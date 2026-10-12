# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sound-event detection for Granary segments.

Framewise AudioSet tagging and event postprocessing run in one stage, so the framewise matrix never leaves memory:

1. PANNs ``Cnn14_DecisionLevelMax`` at 16 kHz (window 1024, hop 320, 64 mel, 50-14000 Hz) produces framewise
   AudioSet probabilities (50 fps), stored as float16; inputs shorter than 10240 samples are
   zero-padded and the output truncated back to the valid frames.
2. Speech probability per frame = noisy-OR over the speech classes, median-filtered (0.2 s -> 11 frames),
   thresholded at 0.2, contiguous runs become events, events closer than 0.1 s (5 frames) are merged.
3. Each event is scored by a gradient-boosted classifier (joblib bundle) on features of the raw framewise slice;
   events with clean-speech probability below 0.11975 are dropped.
4. Weighted speech rate / model confidence / classifier confidence over the manifest segment duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from nemo_curator.stages.audio.tts.audio_io import (
    SegmentAudioError,
    TarAudioConfig,
    iter_segments,
    load_segment_audio,
    make_audio_config,
)
from nemo_curator.stages.audio.tts.audioset_labels import AUDIOSET_LABELS
from nemo_curator.stages.audio.tts.fields import set_dotted
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

DEFAULT_CHECKPOINT_FILENAME = "Cnn14_DecisionLevelMax_mAP=0.385.pth"
DEFAULT_CLASSIFIER_THRESHOLD = 0.11975032188418763

SPEECH_CLASSES: tuple[str, ...] = (
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Speech synthesizer",
)
MUSIC_CLASSES: tuple[str, ...] = (
    "Music", "Musical instrument", "Guitar", "Piano", "Drum kit", "Flute",
    "Saxophone", "Trumpet", "Orchestra", "Choir", "Singing", "Male singing",
    "Female singing", "Child singing", "Rapping", "Humming", "Yodeling",
    "Chant", "Mantra", "Synthetic singing", "Whistling", "Pop music",
    "Rock music", "Jazz", "Classical music", "Electronic music",
    "Hip hop music", "Reggae", "Country", "Blues", "Opera",
    "Ambient music", "Dance music", "Soundtrack music",
)  # fmt: skip
NATURAL_NOISE_CLASSES: tuple[str, ...] = (
    "Wind", "Rustling leaves", "Thunderstorm", "Thunder", "Rain", "Raindrop",
    "Stream", "Waterfall", "Ocean", "Waves, surf", "Fire", "Crackle",
    "Bird", "Crow", "Caw", "Owl", "Hoot", "Frog", "Croak",
    "Insect", "Cricket", "Buzz",
)  # fmt: skip
URBAN_NOISE_CLASSES: tuple[str, ...] = (
    "Traffic noise, roadway noise", "Car", "Car passing by", "Car alarm",
    "Bus", "Truck", "Motorcycle", "Police car (siren)", "Ambulance (siren)",
    "Train", "Train whistle", "Train horn", "Aircraft engine", "Jet engine",
    "Helicopter", "Jackhammer",
)  # fmt: skip
ANIMAL_CLASSES: tuple[str, ...] = (
    "Dog", "Bark", "Howl", "Growling", "Cat", "Meow", "Purr", "Hiss",
    "Horse", "Moo", "Pig", "Oink", "Sheep", "Bleat", "Cluck", "Duck", "Quack",
)  # fmt: skip
IMPULSE_NOISE_CLASSES: tuple[str, ...] = (
    "Explosion", "Gunshot, gunfire", "Machine gun", "Fireworks", "Firecracker",
    "Burst, pop", "Boom", "Bang", "Breaking", "Knock", "Ping", "Ding",
)  # fmt: skip
INDOOR_NOISE_CLASSES: tuple[str, ...] = (
    "Door", "Slam", "Squeak", "Cupboard open or close", "Drawer open or close",
    "Dishes, pots, and pans", "Microwave oven", "Blender", "Toilet flush",
    "Vacuum cleaner", "Hair dryer", "Typing", "Computer keyboard", "Printer",
    "Clock", "Tick", "Tick-tock", "Air conditioning", "Mechanical fan",
)  # fmt: skip
BACKGROUND_NOISE_CLASSES: tuple[str, ...] = (
    "White noise", "Pink noise", "Static", "Mains hum", "Distortion",
    "Reverberation", "Echo", "Vibration",
)  # fmt: skip
VOCAL_NONVERBAL_CLASSES: tuple[str, ...] = (
    "Laughter", "Baby laughter", "Giggle", "Snicker", "Chuckle, chortle",
    "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan",
    "Sigh", "Gasp", "Pant", "Snoring", "Cough", "Sneeze", "Hiccup",
    "Burping, eructation", "Yawn",
)  # fmt: skip
SUPERCLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "speech": SPEECH_CLASSES,
    "music": MUSIC_CLASSES,
    "natural_noise": NATURAL_NOISE_CLASSES,
    "urban_noise": URBAN_NOISE_CLASSES,
    "animal": ANIMAL_CLASSES,
    "impulse_noise": IMPULSE_NOISE_CLASSES,
    "indoor_noise": INDOOR_NOISE_CLASSES,
    "background_noise": BACKGROUND_NOISE_CLASSES,
    "vocal_nonverbal": VOCAL_NONVERBAL_CLASSES,
}
KEY_INDIVIDUAL_CLASSES: tuple[str, ...] = (
    "Speech", "Music", "Singing", "Silence", "Rain", "Bird", "Dog",
    "Engine", "Chatter", "Hubbub, speech noise, speech babble",
)  # fmt: skip
COMPACT_STATS: tuple[str, ...] = ("mean", "std", "p50", "p90", "p95", "max", "frac_gt_0.3")


# ---------------------------------------------------------------------------
# framewise -> speech events
# ---------------------------------------------------------------------------


def make_superclass_probs(
    framewise: np.ndarray, labels: Sequence[str], members: Sequence[str], mode: str = "noisy_or"
) -> np.ndarray:
    """Combine member-class probabilities ``(T, C)`` into one superclass track ``(T,)``."""
    idxs = [labels.index(member) for member in members if member in labels]
    if not idxs:
        msg = "None of superclass member labels were found in labels list."
        raise ValueError(msg)
    probs = framewise[:, idxs].astype(np.float32, copy=False)
    if mode == "max":
        return probs.max(axis=1)
    if mode == "mean":
        return probs.mean(axis=1)
    if mode == "noisy_or":
        return 1.0 - np.prod(1.0 - probs, axis=1)
    msg = f"Unknown mode: {mode}"
    raise ValueError(msg)


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Seconds -> frame count, forced odd (median filter window)."""
    frames = round(seconds * fps)
    if frames % 2 == 0:
        frames += 1
    return max(frames, 1)


def hysteresis_threshold(frame_probs: np.ndarray, low_thr: float = 0.4, high_thr: float = 0.6) -> np.ndarray:
    active = False
    mask = np.zeros_like(frame_probs, dtype=bool)
    for index, prob in enumerate(frame_probs):
        if not active and prob >= high_thr:
            active = True
        elif active and prob < low_thr:
            active = False
        mask[index] = active
    return mask


def mask_to_frame_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Boolean mask -> ``(start_frame, end_frame_exclusive)`` runs."""
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0], strict=True))


Event = tuple[int, int, float, float, float, float]  # start_frame, end_frame (excl.), onset, offset, mean, max


def segments_to_events(
    segments: Sequence[tuple[int, int]], probs: np.ndarray, fps: float, min_dur: float = 0.0
) -> list[Event]:
    """Convert frame segments to timed events with mean/max confidence."""
    events: list[Event] = []
    for start, end in segments:
        if end <= start:
            continue
        onset, offset = start / fps, end / fps
        if (offset - onset) < min_dur:
            continue
        seg = probs[start:end]
        if seg.size == 0:
            continue
        events.append((int(start), int(end), float(onset), float(offset), float(seg.mean()), float(seg.max())))
    return events


def merge_events_with_probs(
    events: Sequence[Event], probs: np.ndarray, fps: float, max_gap: float = 0.2, min_dur: float = 0.0
) -> list[Event]:
    """Merge events whose frame gap is <= ``max_gap`` seconds; mean/max are recomputed from ``probs``."""
    if not events:
        return []
    ordered = sorted(events, key=lambda event: event[0])
    max_gap_frames = max(round(max_gap * fps), 0)
    merged: list[Event] = []

    def flush(start: int, end: int) -> None:
        if end <= start:
            return
        onset, offset = start / fps, end / fps
        if (offset - onset) < min_dur:
            return
        seg = probs[start:end]
        if seg.size == 0:
            return
        merged.append((int(start), int(end), float(onset), float(offset), float(seg.mean()), float(seg.max())))

    cur_start, cur_end = ordered[0][0], ordered[0][1]
    for event in ordered[1:]:
        start, end = event[0], event[1]
        if start - cur_end <= max_gap_frames:
            cur_end = max(cur_end, end)
        else:
            flush(cur_start, cur_end)
            cur_start, cur_end = start, end
    flush(cur_start, cur_end)
    return merged


def class_framewise_to_events(  # noqa: PLR0913
    frame_probs: np.ndarray,
    fps: float,
    thr: float = 0.5,
    min_dur: float = 0.0,
    *,
    enable_smoothing: bool = False,
    smoothing_window_sec: float = 0.2,
    enable_hysteresis: bool = False,
    low_thr: float = 0.4,
    high_thr: float = 0.6,
    enable_merge: bool = False,
    merge_max_gap: float = 0.2,
) -> list[Event]:
    """One-class framewise probabilities -> events (smoothing, threshold, runs, merge)."""
    from scipy.ndimage import median_filter

    probs = frame_probs.astype(np.float32, copy=False)
    if enable_smoothing:
        probs = median_filter(probs, size=seconds_to_frames(smoothing_window_sec, fps))
    mask = hysteresis_threshold(probs, low_thr=low_thr, high_thr=high_thr) if enable_hysteresis else probs >= thr
    events = segments_to_events(mask_to_frame_segments(mask), probs=probs, fps=fps, min_dur=min_dur)
    if enable_merge and events:
        events = merge_events_with_probs(events, probs=probs, fps=fps, max_gap=merge_max_gap, min_dur=min_dur)
    return events


# ---------------------------------------------------------------------------
# event features + gradient-boosted classifier
# ---------------------------------------------------------------------------


def get_class_indices(labels: Sequence[str], class_names: Sequence[str]) -> list[int]:
    label_to_idx = {label: index for index, label in enumerate(labels)}
    return [label_to_idx[name] for name in class_names if name in label_to_idx]


def compute_stats(values: np.ndarray) -> dict[str, float]:
    """Summary statistics (mean, std, percentiles, max, fraction >= 0.3) of a frame series."""
    if values.size == 0:
        return dict.fromkeys(COMPACT_STATS, float("nan"))
    v = values.astype(np.float64)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p50": float(np.quantile(v, 0.50)),
        "p90": float(np.quantile(v, 0.90)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
        "frac_gt_0.3": float(np.mean(values >= 0.3)),  # noqa: PLR2004
    }


def _group_features(seg: np.ndarray, group_name: str, idxs: list[int]) -> dict[str, float]:
    if not idxs:
        return {f"{group_name}_max_{stat}": float("nan") for stat in COMPACT_STATS}
    features = {f"{group_name}_max_{stat}": value for stat, value in compute_stats(seg[:, idxs].max(axis=1)).items()}
    if group_name == "speech":
        noisy_or = 1.0 - np.prod(1.0 - seg[:, idxs].astype(np.float64), axis=1)
        features.update(
            {f"speech_nor_{stat}": value for stat, value in compute_stats(noisy_or.astype(np.float32)).items()}
        )
    return features


def extract_features_for_segment(  # noqa: PLR0913
    framewise: np.ndarray,
    fps: float,
    labels: Sequence[str],  # noqa: ARG001
    start_time: float,
    end_time: float,
    label_to_idx: dict[str, int],
    group_indices: dict[str, list[int]],
) -> dict[str, float]:
    """Statistics of the raw framewise slice ``[floor(start*fps), ceil(end*fps))`` of one event."""
    start = max(0, min(len(framewise), int(np.floor(start_time * fps))))
    end = max(0, min(len(framewise), int(np.ceil(end_time * fps))))
    if end <= start:
        return {}
    seg = framewise[start:end]
    features: dict[str, float] = {"duration": end_time - start_time, "n_frames": end - start}
    for group_name, idxs in group_indices.items():
        features.update(_group_features(seg, group_name, idxs))
    for cls in KEY_INDIVIDUAL_CLASSES:
        if cls in label_to_idx:
            safe = cls.replace(" ", "_").replace(",", "").replace("-", "_")
            for stat, value in compute_stats(seg[:, label_to_idx[cls]]).items():
                features[f"class_{safe}_{stat}"] = value
    return features


def load_gbm_bundle(path: str | Path) -> dict[str, Any]:
    """Load the joblib classifier bundle ``{model, feature_columns, multiclass, clean_class_id, ...}``."""
    import joblib

    bundle = joblib.load(str(Path(path).expanduser()))
    missing = [key for key in ("model", "feature_columns") if key not in bundle]
    if not isinstance(bundle, dict) or missing:
        msg = f"GBM bundle {path} must be a dict with 'model' and 'feature_columns' (missing {missing})"
        raise ValueError(msg)
    return bundle


def classify_events(  # noqa: PLR0913
    predicted_events: list[dict[str, Any]],
    framewise: np.ndarray,
    labels: Sequence[str],
    fps: float,
    bundle: dict[str, Any],
    threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
) -> list[dict[str, Any]]:
    """Attach ``classifier_prob`` (clean-speech probability) to each event; keep events with prob >= threshold.

    Features missing from the extracted dict are fed as 0.0 and NaN/inf are zeroed.
    """
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    is_multiclass = bundle.get("multiclass", False)
    clean_class_id = bundle.get("clean_class_id", 0)
    label_to_idx = {label: index for index, label in enumerate(labels)}
    group_indices = {name: get_class_indices(labels, classes) for name, classes in SUPERCLASS_GROUPS.items()}
    class_idx = list(model.named_steps["clf"].classes_).index(clean_class_id) if is_multiclass else 1
    kept = []
    for event in predicted_events:
        features = extract_features_for_segment(
            framewise,
            fps,
            labels,
            float(event["start_time"]),
            float(event["end_time"]),
            label_to_idx,
            group_indices,
        )
        if not features:
            continue
        x = np.array([[features.get(column, 0.0) for column in feature_columns]], dtype=np.float32)
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        prob = float(model.predict_proba(x)[0, class_idx])
        event["classifier_prob"] = prob
        if prob >= threshold:
            kept.append(event)
    return kept


# ---------------------------------------------------------------------------
# segment annotation
# ---------------------------------------------------------------------------


def empty_annotation(error: str | None = None) -> dict[str, Any]:
    """Failed segments get zero scores so they fail the downstream SED filter."""
    return {
        "predicted_events": [],
        "error": error,
        "weighted_speech_rate": 0.0,
        "weighted_model_confidence": 0.0,
        "weighted_classifier_confidence": 0.0,
    }


def weighted_summary(events: Sequence[dict[str, Any]], duration: float) -> dict[str, float]:
    """Event coverage and confidences weighted by each event's share of the whole segment duration."""
    if duration <= 0:
        return {"weighted_speech_rate": 0.0, "weighted_model_confidence": 0.0, "weighted_classifier_confidence": 0.0}
    speech_duration = model_confidence = classifier_confidence = 0.0
    for event in events:
        start = float(event.get("start_time", 0.0) or 0.0)
        end = float(event.get("end_time", 0.0) or 0.0)
        event_duration = max(0.0, min(end, duration) - max(start, 0.0))
        if event_duration <= 0:
            continue
        weight = event_duration / duration
        speech_duration += event_duration
        model_confidence += weight * float(event.get("mean_confidence", 0.0) or 0.0)
        classifier_confidence += weight * float(event.get("classifier_prob", 0.0) or 0.0)
    return {
        "weighted_speech_rate": speech_duration / duration,
        "weighted_model_confidence": model_confidence,
        "weighted_classifier_confidence": classifier_confidence,
    }


@dataclass(frozen=True)
class SpeechEventConfig:
    """Speech-event detection and classifier settings."""

    speech_class_names: tuple[str, ...] = SPEECH_CLASSES
    unified_speech_label: str = "Speech"
    superclass_probs_mode: str = "noisy_or"
    threshold: float = 0.2
    min_duration: float = 0.0
    enable_smoothing: bool = True
    smoothing_window_sec: float = 0.2
    enable_hysteresis: bool = False
    enable_merge: bool = True
    merge_max_gap: float = 0.1
    classifier_threshold: float = DEFAULT_CLASSIFIER_THRESHOLD


def detect_speech_events(
    framewise: np.ndarray, fps: float, labels: Sequence[str], config: SpeechEventConfig | None = None
) -> list[dict[str, Any]]:
    config = config or SpeechEventConfig()
    frame_probs = make_superclass_probs(framewise, labels, config.speech_class_names, config.superclass_probs_mode)
    events = class_framewise_to_events(
        frame_probs,
        fps,
        thr=config.threshold,
        min_dur=config.min_duration,
        enable_smoothing=config.enable_smoothing,
        smoothing_window_sec=config.smoothing_window_sec,
        enable_hysteresis=config.enable_hysteresis,
        enable_merge=config.enable_merge,
        merge_max_gap=config.merge_max_gap,
    )
    return [
        {
            "label": config.unified_speech_label,
            "start_time": event[2],
            "end_time": event[3],
            "mean_confidence": event[4],
            "max_confidence": event[5],
        }
        for event in events
    ]


def annotate_framewise(  # noqa: PLR0913
    framewise: np.ndarray,
    fps: float,
    duration: float,
    *,
    labels: Sequence[str] = AUDIOSET_LABELS,
    valid_frames: int | None = None,
    bundle: dict[str, Any] | None = None,
    config: SpeechEventConfig | None = None,
) -> dict[str, Any]:
    """Framewise ``(T, 527)`` probabilities -> SED annotation (events + weighted summary)."""
    config = config or SpeechEventConfig()
    if valid_frames is not None:
        framewise = framewise[: int(valid_frames)]
    labels = list(labels)
    events = detect_speech_events(framewise, fps, labels, config)
    if bundle is not None:
        events = classify_events(events, framewise, labels, fps, bundle, config.classifier_threshold)
    annotation: dict[str, Any] = {"predicted_events": events, "error": None}
    annotation.update(weighted_summary(events, duration))
    return annotation


@dataclass
class SegmentSEDStage(ProcessingStage[AudioTask, AudioTask]):
    """Annotate every segment with CNN14 speech events, GBM clean-speech scores and weighted SED summaries.

    Writes ``output_key`` = ``{predicted_events, weighted_speech_rate, weighted_model_confidence,
    weighted_classifier_confidence, error}`` per segment; rows and segments are never dropped. Inference runs one
    segment at a time because zero-padding a ragged batch changes the tail frames.

    Args:
        checkpoint_path: ``Cnn14_DecisionLevelMax_mAP=0.385.pth``; ``None`` uses (and on first use downloads into)
            the PyTorch Hub checkpoint dir.
        classifier_path: joblib GBM bundle (``gbt_multiclass_model.joblib``). Required unless
            ``enable_classifier=False``; without the classifier events carry no ``classifier_prob``.
        framewise_key: optional segment key to also store the float16 framewise matrix (``ndarray``) under.
    """

    name: str = "SegmentSED"
    output_key: str = "sed"
    checkpoint_path: str | None = None
    classifier_path: str | None = None
    enable_classifier: bool = True
    model_type: str = "Cnn14_DecisionLevelMax"
    sample_rate: int = 16000
    window_size: int = 1024
    hop_size: int = 320
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 14000
    classes_num: int = 527
    pad_short_segments: bool = True
    framewise_dtype: str = "float16"
    framewise_key: str | None = None
    detection: SpeechEventConfig = field(default_factory=SpeechEventConfig)
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    audio_source: str = "original"
    enhanced_key: str = "enhanced_audio"
    enhanced_fallback: bool = False
    device: str | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.25))
    batch_size: int = 1
    _model: Any = field(default=None, init=False, repr=False)
    _torch_device: Any = field(default=None, init=False, repr=False)
    _bundle: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        if self.enable_classifier and not self.classifier_path:
            msg = "SegmentSEDStage needs classifier_path (GBM joblib bundle) unless enable_classifier=False"
            raise ValueError(msg)
        if self.framewise_dtype not in ("float16", "float32"):
            msg = f"framewise_dtype must be 'float16' or 'float32', got {self.framewise_dtype!r}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key.split(".")[0]]

    @property
    def min_input_samples(self) -> int:
        return max(self.window_size, self.hop_size * 32)

    @property
    def fps(self) -> float:
        return float(self.sample_rate // self.hop_size)

    def _checkpoint(self) -> Path:
        import torch

        if self.checkpoint_path:
            return Path(self.checkpoint_path).expanduser()
        return Path(torch.hub.get_dir()) / "checkpoints" / DEFAULT_CHECKPOINT_FILENAME

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Fetch the public PANNs checkpoint when missing (the 32 kHz adapter only validates the download)."""
        checkpoint = self._checkpoint()
        if checkpoint.is_file():
            return
        if checkpoint.name != DEFAULT_CHECKPOINT_FILENAME:
            msg = f"CNN14 checkpoint {checkpoint} does not exist"
            raise FileNotFoundError(msg)
        from nemo_curator.models.sed.panns import PANNsSEDAdapter

        PANNsSEDAdapter(checkpoint_path=str(checkpoint)).download_weights_on_node()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        import torch

        from nemo_curator.models.sed import get_model_class

        if self._model is None:
            checkpoint = self._checkpoint()
            if not checkpoint.is_file():
                self.setup_on_node()
            model = get_model_class(self.model_type)(
                sample_rate=self.sample_rate,
                window_size=self.window_size,
                hop_size=self.hop_size,
                mel_bins=self.mel_bins,
                fmin=self.fmin,
                fmax=self.fmax,
                classes_num=self.classes_num,
            )
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
            self._model = model
        self._torch_device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model.to(self._torch_device)
        self._model.eval()
        if self.enable_classifier and self._bundle is None:
            self._bundle = load_gbm_bundle(str(self.classifier_path))

    def teardown(self) -> None:
        self._model = None
        self._bundle = None

    def infer_framewise(self, waveform: np.ndarray) -> tuple[np.ndarray, int]:
        """Mono 16 kHz waveform -> ``(framewise[T, 527], valid_frames)`` (float16 by default)."""
        import torch

        original_num_samples = int(waveform.shape[0])
        if self.pad_short_segments and original_num_samples < self.min_input_samples:
            waveform = np.pad(waveform, (0, self.min_input_samples - original_num_samples), mode="constant")
        x = torch.from_numpy(np.asarray(waveform, dtype=np.float32)[None, :]).float().to(self._torch_device)
        with torch.no_grad():
            framewise = self._model(x, None)["framewise_output"].detach().cpu().numpy()[0]
        framewise = framewise.astype(np.float16 if self.framewise_dtype == "float16" else np.float32)
        valid_frames = min(int(np.ceil(original_num_samples / self.hop_size)), int(framewise.shape[0]))
        return framewise, valid_frames

    def annotate_segment(self, segment: dict[str, Any], audio_config: TarAudioConfig) -> dict[str, Any]:
        try:
            waveform, _ = load_segment_audio(segment, self.sample_rate, audio_config)
        except SegmentAudioError as exc:
            return empty_annotation(exc.code)
        try:
            framewise, valid_frames = self.infer_framewise(waveform)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CNN14 inference failed: {}", exc)
            return empty_annotation(repr(exc))
        if self.framewise_key:
            set_dotted(segment, self.framewise_key, framewise[:valid_frames])
        was_padded = self.pad_short_segments and int(waveform.shape[0]) < self.min_input_samples
        try:
            return annotate_framewise(
                framewise,
                self.fps,
                float(segment.get("duration", 0.0) or 0.0),
                valid_frames=valid_frames if was_padded else None,
                bundle=self._bundle,
                config=self.detection,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SED postprocessing failed: {}", exc)
            return empty_annotation(repr(exc))

    def process(self, task: AudioTask) -> AudioTask:
        if self._model is None:
            self.setup()
        audio_config = make_audio_config(self)
        for _, segment in iter_segments(task.data):
            set_dotted(segment, self.output_key, self.annotate_segment(segment, audio_config))
        return task
