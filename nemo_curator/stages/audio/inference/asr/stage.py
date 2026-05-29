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

"""Generic audio ASR Curator stage.

Implements the SDP-V2 stage half of the stage-adapter split (see the
design doc, "Replaceability - Stage -> Adapter" section). The stage owns
exactly the Curator-side glue:

* validates ``task.data`` against ``inputs()`` / ``outputs()``;
* unpacks per-task knobs (waveform, sample rate, ISO language code -> name)
  into the dicts the adapter expects;
* dispatches the single ``self._adapter.transcribe_batch(items)`` call;
* writes the predicted text(s) onto ``task.data`` under stage-configured
  output keys;
* marks ``_skip_me`` for adapter-flagged skips;
* drops the in-memory waveform after inference (unless ``keep_waveform``);
* emits performance metrics in the shape ``perf_summary_merged.json``
  consumers already expect.

The stage knows nothing about which model is running. The concrete model
class is resolved at runtime from the YAML's ``adapter_target`` string via
``hydra.utils.get_class`` (same pattern Curator's framework uses today in
``nemo_curator/config/run.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import hydra.utils
from loguru import logger

from nemo_curator.adapters.asr.base import ASRAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


# Per the design doc, the ISO-code -> human-readable-language-name mapping
# lives on the stage (Tier-3 per-task knob extraction). The adapter receives
# already-resolved names like "English", "Spanish" via the item dict.
_LANG_CODE_TO_NAME: dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "mt": "Maltese",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


@dataclass
class ASRStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio speech-recognition Curator stage with pluggable adapter.

    Args:
        adapter_target: Tier-1 swap surface. Fully-qualified class path
            of the concrete :class:`~nemo_curator.adapters.asr.ASRAdapter`
            implementation (e.g.
            ``"nemo_curator.adapters.asr.QwenOmniASRAdapter"``). Resolved
            at ``setup()`` time via ``hydra.utils.get_class``.
        model_id: Tier-1. Model checkpoint identifier, forwarded both to
            ``ASRAdapter.prefetch_weights`` (in ``setup_on_node``) and to
            the adapter constructor.
        revision: Tier-1. Optional model revision to pin.
        waveform_key / sample_rate_key: Keys into ``task.data`` for the
            input waveform numpy array and its sample rate.
        source_lang_key: Key into ``task.data`` carrying an ISO language
            code. Mapped to a human-readable name before being forwarded
            to the adapter.
        default_language: Fallback ISO code / language name used when a
            task has no ``source_lang_key`` value.
        pred_text_key: Key under which the Turn-1 prediction is written.
        secondary_text_key: When set, enables Turn-2 / refined output -
            the stage writes ``ASRResult.secondary_text or ""`` under
            this key, and reports the key in ``outputs()``. ``None``
            means single-turn semantics (the adapter's secondary text is
            ignored even if produced).
        skip_me_key: Key set to ``"empty_audio"`` for adapter-flagged
            skipped items (matches the pre-split convention).
        keep_waveform: When False (default), drops
            ``task.data[waveform_key]`` after inference to keep the
            downstream task payload small.
        prefetch_fail_on_error: When False, ``setup_on_node`` warns and
            defers weight prefetch to ``setup()`` instead of raising.
        adapter_kwargs: Tier-2. Opaque dict forwarded to the adapter
            constructor as ``**adapter_kwargs``. The stage NEVER reads
            inside this dict - it is the adapter's private knob bag.
        resources / batch_size: Standard Curator stage knobs.
    """

    name: str = "ASR_inference"

    # ---- Tier 1: swap surface ----
    adapter_target: str = ""
    model_id: str = ""
    revision: str | None = None

    # ---- Tier 1: universal stage knobs ----
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    source_lang_key: str = "source_lang"
    default_language: str | None = None
    pred_text_key: str = "pred_text"
    secondary_text_key: str | None = None
    skip_me_key: str = "_skip_me"
    keep_waveform: bool = False
    prefetch_fail_on_error: bool = True

    # ---- Tier 2: opaque adapter knob bag ----
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---- Standard Curator stage knobs ----
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 32

    def __post_init__(self) -> None:
        if not self.adapter_target:
            msg = (
                "ASRStage.adapter_target is required - set it in YAML to a fully-qualified "
                "adapter class path (e.g. 'nemo_curator.adapters.asr.QwenOmniASRAdapter')."
            )
            raise ValueError(msg)
        self._adapter: ASRAdapter | None = None

    # ------------------------------------------------------------------
    # Adapter lifecycle - thin wrappers around the doc-prescribed shape:
    #   cls = hydra.utils.get_class(self.adapter_target)
    #   self._adapter = cls(model_id=..., revision=..., **self.adapter_kwargs)
    # ------------------------------------------------------------------

    def _adapter_class(self) -> type:
        return hydra.utils.get_class(self.adapter_target)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Cache model weights once per node (no GPU allocation).

        Worker processes do not preserve ``_adapter`` across pickle, so
        ``setup()`` re-creates the adapter on each worker.  This method
        only ensures the snapshot is cached on local storage so that
        ``setup()`` doesn't trigger parallel downloads on multi-GPU nodes.
        """
        try:
            prefetch_t0 = time.perf_counter()
            self._adapter_class().prefetch_weights(self.model_id, self.revision)
            logger.info(
                "ASR weights cached on node for {} ({}) in {:.3f}s",
                self.model_id,
                self.adapter_target,
                time.perf_counter() - prefetch_t0,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"ASRStage: prefetch_weights failed for {self.model_id}"
            if self.prefetch_fail_on_error:
                raise RuntimeError(msg) from exc
            logger.warning("{}; setup() will retry: {}", msg, exc)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._adapter is None:
            cls = self._adapter_class()
            self._adapter = cls(
                model_id=self.model_id,
                revision=self.revision,
                **self.adapter_kwargs,
            )
            self._adapter.setup()
            logger.info("ASR adapter ready on worker ({})", self.adapter_target)

    def teardown(self) -> None:
        if self._adapter is not None:
            self._adapter.teardown()
            self._adapter = None

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        keys = [self.pred_text_key, self.skip_me_key]
        if self.secondary_text_key:
            keys.append(self.secondary_text_key)
        return [], keys

    # ------------------------------------------------------------------
    # Per-task language resolution (ISO code -> human name)
    # ------------------------------------------------------------------

    def _resolve_language(self, task: AudioTask) -> str | None:
        code = task.data.get(self.source_lang_key) if self.source_lang_key else None
        if code:
            return _LANG_CODE_TO_NAME.get(code, code)
        if self.default_language:
            return _LANG_CODE_TO_NAME.get(self.default_language, self.default_language)
        return None

    def _build_items(self, tasks: list[AudioTask]) -> list[dict[str, Any]]:
        return [
            {
                "waveform": t.data[self.waveform_key],
                "sample_rate": t.data[self.sample_rate_key],
                "language": self._resolve_language(t),
                "task_id": t.task_id,
            }
            for t in tasks
        ]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        msg = f"{type(self).__name__} only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not tasks:
            return []
        for task in tasks:
            if not self.validate_input(task):
                msg = (
                    f"Task {task.task_id} missing required columns for "
                    f"{type(self).__name__}: {self.inputs()}"
                )
                raise ValueError(msg)

        if self._adapter is None:
            msg = "Adapter not initialized - setup() was not called"
            raise RuntimeError(msg)

        items = self._build_items(tasks)
        t0 = time.perf_counter()
        results = self._adapter.transcribe_batch(items)
        inference_elapsed = time.perf_counter() - t0
        model_metrics = dict(getattr(self._adapter, "last_metrics", {}) or {})

        if len(results) != len(tasks):
            msg = (
                f"Adapter {self.adapter_target} returned {len(results)} results "
                f"for {len(tasks)} tasks (must match 1:1)"
            )
            raise RuntimeError(msg)

        skipped_count = 0
        for task, item, result in zip(tasks, items, results, strict=True):
            task.data[self.pred_text_key] = result.text
            if self.secondary_text_key:
                task.data[self.secondary_text_key] = result.secondary_text or ""
            if result.skipped:
                task.data[self.skip_me_key] = "empty_audio"
                skipped_count += 1
            if not self.keep_waveform:
                task.data.pop(self.waveform_key, None)

        # ----- metrics (shape preserved from pre-split InferenceQwenOmniStage
        #       so existing perf_summary_merged.json consumers keep working) ----
        waveforms = [it["waveform"] for it in items]
        sample_rates = [it["sample_rate"] for it in items]
        metrics: dict[str, float] = {
            "utterances_input": float(len(tasks)),
            "utterances_processed": float(max(0, len(tasks) - skipped_count)),
            "utterances_skipped": float(skipped_count),
            "audio_duration_s": sum(
                float(w.shape[0]) / float(sr)
                for w, sr in zip(waveforms, sample_rates, strict=False)
                if sr and w is not None and getattr(w, "size", 0) > 0
            ),
            "waveform_bytes": sum(
                float(getattr(w, "nbytes", 0)) for w in waveforms if w is not None
            ),
            "output_chars": float(
                sum(len(r.text) for r in results)
                + sum(len(r.secondary_text or "") for r in results)
            ),
            "output_tokens": float(model_metrics.get("output_tokens", 0.0)),
            "turn1_output_tokens": float(model_metrics.get("turn1_output_tokens", 0.0)),
            "turn2_output_tokens": float(model_metrics.get("turn2_output_tokens", 0.0)),
            "inference_time_s": inference_elapsed,
        }
        # A5-fix preserved: pass through adapter-side scalar metrics under a
        # "model_<name>" alias, but skip aliases that would just restate a key
        # the stage already emits above.
        metrics.update({
            f"model_{name}": value
            for name, value in model_metrics.items()
            if isinstance(value, (int, float)) and name not in metrics
        })
        self._log_metrics(metrics)

        if skipped_count:
            logger.info(
                f"ASRStage ({self.adapter_target}): marked {skipped_count}/{len(tasks)} "
                f"tasks as empty_audio ({self.skip_me_key})",
            )
        logger.info(
            f"ASRStage ({self.adapter_target}): generated {len(results)} predictions "
            f"(secondary_text={'on' if self.secondary_text_key else 'off'})",
        )
        return tasks
