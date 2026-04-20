#!/usr/bin/env python3
"""
Batch QA checks for multi-speaker audio sessions.

All sessions are processed stage-by-stage (not session-by-session) to
maximise GPU utilisation and reduce model load overhead.

Pipeline stages:
  Stage 1 — WER: batch-transcribe all channel audio with Parakeet-TDT-0.6B-v3
            (batch_size=10), then evaluate per-channel WER via meeteval-wer.
  Stage 2 — DER: batch-diarise all audio (mixed + per-channel) with Sortformer,
            then run Silero VAD sequentially; union SAD → evaluate DER.
  Stage 3 — AEC (optional, pass --check_aec): measure echo reduction per
            channel (informational only, does not affect PASS/FAIL).
  Stage 4 — SNR: silence power + RTTM-based SNR_VAD per channel.

Usage:
    python run_qa_checks.py --file_list file_list.json --output qa_results.json
    python run_qa_checks.py --file_list file_list.json --output qa_results.json --check_aec

Each session in file_list.json must include:

- ``seglsts``: channel id → path to that channel's ``.seglst.json`` reference
  (same keys as ``channels``).
- ``sample_rate``: integer Hz for **all** WAVs in that session (channels + mixed);
  file SR may differ and will be resampled to this value.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, "/home/taejinp/projects/AEC2ch/AEC2ch/src")
from aec2ch import AEC2ch
from aec2ch.evaluate import silence_mask, rms_power_db

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

print(">>> script starting, imports done", flush=True)

# ---------------------------------------------------------------------------
# Pass/fail thresholds (one threshold per metric, shared across all channels)
# A session FAILs if ANY channel exceeds ANY threshold.
# ---------------------------------------------------------------------------
DER_MIX_MAX = 0.05        # mixed-signal DER ≤ 5 %
DER_CH_MAX = 0.10         # per-channel DER ≤ 10 %
SNR_MIN_DB = 20.0         # RTTM-based SNR ≥ 20 dB
ECHO_RED_MAX_DB = 10.0    # AEC echo reduction ≤ 10 dB (high = heavy bleed)
SILENCE_RMS_MAX_DB = -40.0  # silence floor ≤ −40 dB
WER_MAX = 0.10              # per-channel WER ≤ 10 %


# ═══════════════════════════════════════════════════════════════════════════
# Audio sample rate (declared per session in file_list.json → "sample_rate")
# ═══════════════════════════════════════════════════════════════════════════


def _load_mono_wav_at_declared_sr(path: str, declared_sr: int) -> np.ndarray:
    """
    Load mono float32 audio and resample to ``declared_sr`` if the file differs.
    RTTM / seglst timings are interpreted in seconds; sample indices use declared_sr.
    """
    import torch
    import torchaudio

    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w[:, 0]
    if sr == declared_sr:
        return w
    log.warning(
        f"Resampling {path}: file SR={sr} Hz → manifest sample_rate={declared_sr} Hz"
    )
    t = torch.from_numpy(w.copy()).float().unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, declared_sr)
    return t.squeeze(0).numpy()


def _wav_tensor_for_silero_vad(wav_np: np.ndarray, declared_sr: int):
    """
    Silero VAD expects 8000 or 16000 Hz. Resample from declared_sr when needed.
    Returns (waveform tensor, sampling_rate for get_speech_timestamps).
    """
    import torch
    import torchaudio

    wav = torch.from_numpy(wav_np).float()
    silero_sr = 8000 if declared_sr == 8000 else 16000
    if declared_sr != silero_sr:
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), declared_sr, silero_sr
        ).squeeze(0)
    return wav, silero_sr


def _parse_sample_rate(entry: dict) -> int:
    if "sample_rate" not in entry:
        raise ValueError(
            f"session {entry.get('session_id', '?')!r}: missing required "
            f'integer field "sample_rate" (Hz for all WAVs in this session)'
        )
    sr = int(entry["sample_rate"])
    if sr <= 0:
        raise ValueError(
            f"session {entry.get('session_id', '?')!r}: sample_rate must be positive, got {sr}"
        )
    return sr


def _warn_if_wav_sr_mismatch(path: str, declared_sr: int) -> None:
    try:
        info = sf.info(path)
    except OSError as e:
        log.warning(f"Could not inspect {path}: {e}")
        return
    if info.samplerate != declared_sr:
        log.warning(
            f"{path}: file SR={info.samplerate} Hz ≠ manifest sample_rate={declared_sr} Hz "
            f"(will resample to {declared_sr} Hz for AEC/SNR/Silero)"
        )


def _validate_manifest_sample_rates(manifest: list[dict]) -> None:
    for entry in manifest:
        sr = _parse_sample_rate(entry)
        for _, path in entry.get("channels", {}).items():
            _warn_if_wav_sr_mismatch(path, sr)
        mw = entry.get("mixed_wav")
        if mw:
            _warn_if_wav_sr_mismatch(mw, sr)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class WERResult:
    per_channel_wer: dict[str, float] = field(default_factory=dict)


@dataclass
class DERResult:
    mixed_der: Optional[float] = None
    per_channel_der: dict[str, float] = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


@dataclass
class ChannelBleedResult:
    per_channel_echo_reduction_db: dict[str, float] = field(default_factory=dict)


@dataclass
class SNRResult:
    per_channel_silence_rms_db: dict[str, float] = field(default_factory=dict)
    per_channel_snr_db: dict[str, float] = field(default_factory=dict)


@dataclass
class SessionQAResult:
    session_id: str = ""
    wer: WERResult = field(default_factory=WERResult)
    der: DERResult = field(default_factory=DERResult)
    channel_bleed: ChannelBleedResult = field(default_factory=ChannelBleedResult)
    snr: SNRResult = field(default_factory=SNRResult)
    passed: bool = True
    fail_reasons: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# RTTM parsing
# ═══════════════════════════════════════════════════════════════════════════
def parse_rttm(rttm_path: str) -> list[tuple[float, float]]:
    """Return list of (onset_sec, offset_sec) from an RTTM file."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or parts[0] != "SPEAKER":
                continue
            onset = float(parts[3])
            dur = float(parts[4])
            segments.append((onset, onset + dur))
    return segments


def merge_segments(
    segments: list[tuple[float, float]], collar: float = 0.0
) -> list[tuple[float, float]]:
    if not segments:
        return []
    segs = sorted(segments)
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s <= merged[-1][1] + collar:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def segments_to_mask(
    segments: list[tuple[float, float]], duration_sec: float, sr: int
) -> np.ndarray:
    n_samples = int(duration_sec * sr)
    mask = np.zeros(n_samples, dtype=bool)
    for s, e in segments:
        i0 = int(s * sr)
        i1 = min(int(e * sr), n_samples)
        mask[i0:i1] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# Check 1 — WER (Parakeet-TDT-0.6B-v3 as checker model)
# ═══════════════════════════════════════════════════════════════════════════

# ---------- Parakeet ASR model ----------
_parakeet_model = None


def _get_parakeet_model():
    global _parakeet_model
    if _parakeet_model is None:
        import nemo.collections.asr as nemo_asr
        t0 = time.time()
        log.info("    [WER] loading parakeet-tdt-0.6b-v3 …")
        _parakeet_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        _parakeet_model.eval()
        log.info(f"    [WER] model loaded in {time.time()-t0:.1f}s")
    return _parakeet_model


def _normalize_hyp(text: str) -> str:
    """Lowercase and strip all punctuation from ASR hypothesis."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return " ".join(text.split())


def _run_meeteval_wer(ref_seglst: str, hyp_seglst: str) -> dict:
    """Run meeteval-wer wer and return the per-reco result dict."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix="_per_reco.json", delete=False) as pf, \
         tempfile.NamedTemporaryFile(suffix="_avg.json", delete=False) as af:
        per_reco_path, avg_path = pf.name, af.name

    cmd = [
        "meeteval-wer", "wer",
        "-r", ref_seglst, "-h", hyp_seglst,
        "--normalizer", "chime8",
        "--per-reco-out", per_reco_path,
        "--average-out", avg_path,
    ]
    log.info(f"    [WER] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error(f"    [WER] meeteval-wer failed: {proc.stderr}")
        os.remove(per_reco_path)
        os.remove(avg_path)
        return {}
    for line in proc.stderr.splitlines():
        if line.strip():
            log.info(f"    [WER] {line.strip()}")

    with open(per_reco_path) as f:
        per_reco = json.load(f)
    os.remove(per_reco_path)
    os.remove(avg_path)

    if len(per_reco) == 1:
        return next(iter(per_reco.values()))
    return per_reco


def check_wer(
    session_id: str,
    channels: dict[str, str],
    seglst_path: str | None,
) -> WERResult:
    """Legacy single-session WER (unused in batch mode, kept for reference)."""
    raise NotImplementedError("Use batch_wer() instead")


# ═══════════════════════════════════════════════════════════════════════════
# Check 2 — DER per channel (Sortformer ∩ Silero VAD vs RTTM, meeteval md_eval_22)
# ═══════════════════════════════════════════════════════════════════════════

# ---------- Sortformer (diar) ----------
_sortformer_model = None


def _get_sortformer_model():
    global _sortformer_model
    if _sortformer_model is None:
        from nemo.collections.asr.models import SortformerEncLabelModel
        t0 = time.time()
        log.info("    [sortformer] loading model …")
        _sortformer_model = SortformerEncLabelModel.from_pretrained(
            "nvidia/diar_streaming_sortformer_4spk-v2.1"
        )
        _sortformer_model.eval()
        _sortformer_model.sortformer_modules.chunk_len = 340
        _sortformer_model.sortformer_modules.chunk_right_context = 40
        _sortformer_model.sortformer_modules.fifo_len = 40
        _sortformer_model.sortformer_modules.spkcache_update_period = 300
        log.info(f"    [sortformer] model loaded in {time.time()-t0:.1f}s")
    return _sortformer_model


def _run_sortformer(audio_path: str) -> list[str]:
    """
    Run Sortformer diarizer on an audio file.
    Returns the raw segment list: each element is 'start end speaker_label'.
    """
    model = _get_sortformer_model()
    t0 = time.time()
    predicted_segments = model.diarize(audio=[audio_path], batch_size=1)
    log.info(f"    [sortformer] diarization done in {time.time()-t0:.1f}s, "
             f"{len(predicted_segments[0])} raw segments")
    return predicted_segments[0]


# ---------- Silero VAD ----------
_silero_model = None


def _get_silero_model():
    global _silero_model
    if _silero_model is None:
        import torch
        t0 = time.time()
        log.info("    [silero] loading model …")
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad",
            trust_repo=True, onnx=False,
        )
        _silero_model._utils = _silero_utils
        log.info(f"    [silero] model loaded in {time.time()-t0:.1f}s")
    return _silero_model


def _run_silero_vad(audio_path: str, declared_sample_rate: int) -> list[str]:
    """
    Run Silero VAD on an audio file.
    Audio is loaded at ``declared_sample_rate`` (manifest), then resampled
    inside Silero to 8 kHz or 16 kHz as required by the model.
    Returns segments as list of 'start end speech' strings (same format as
    Sortformer output, with a single 'speech' label).
    """
    model = _get_silero_model()
    get_speech_timestamps = model._utils[0]

    wav_np = _load_mono_wav_at_declared_sr(audio_path, declared_sample_rate)
    wav, silero_sr = _wav_tensor_for_silero_vad(wav_np, declared_sample_rate)

    t0 = time.time()
    timestamps = get_speech_timestamps(wav, model, sampling_rate=silero_sr,
                                       return_seconds=True)
    log.info(f"    [silero] VAD done in {time.time()-t0:.1f}s, "
             f"{len(timestamps)} segments")

    segments: list[str] = []
    for ts in timestamps:
        segments.append(f"{ts['start']:.3f} {ts['end']:.3f} speech")
    return segments


def _segs_str_to_tuples(segs: list[str]) -> list[tuple[float, float]]:
    """Convert 'start end label' strings to (start, end) tuples."""
    out: list[tuple[float, float]] = []
    for s in segs:
        parts = s.split()
        out.append((float(parts[0]), float(parts[1])))
    return out


# ---------- Unified SAD runner (Sortformer ∪ Silero) ----------
def _run_sad(audio_path: str, declared_sample_rate: int) -> list[str]:
    """
    Run both Sortformer and Silero VAD, return the union of their outputs.
    """
    raw_sortformer = _run_sortformer(audio_path)
    raw_silero = _run_silero_vad(audio_path, declared_sample_rate)

    sf_merged = merge_segments(sorted(_segs_str_to_tuples(raw_sortformer)))
    si_merged = merge_segments(sorted(_segs_str_to_tuples(raw_silero)))

    unioned = merge_segments(sf_merged + si_merged)
    log.info(f"    [SAD] Sortformer={len(sf_merged)} segs, Silero={len(si_merged)} segs "
             f"→ union={len(unioned)}")

    return [f"{s:.3f} {e:.3f} speech" for s, e in unioned]


# ---------- RTTM writer ----------
def _write_rttm(segments: list[str], out_path: str, file_id: str):
    """
    Write segments to RTTM format.
    Each segment is 'start end speaker_label'.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for seg in segments:
            parts = seg.split()
            start, end, spk = float(parts[0]), float(parts[1]), parts[2]
            dur = end - start
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    log.info(f"    [RTTM] wrote {len(segments)} segments → {out_path}")


def _flatten_rttm_to_single_speaker(in_path: str, out_path: str):
    """
    Read an RTTM, merge all speakers into one, merge overlapping segments.
    This turns a diarization RTTM into a SAD RTTM for pure speech/non-speech
    evaluation (no speaker confusion).
    """
    segs = parse_rttm(in_path)
    merged = merge_segments(segs)
    with open(in_path) as f:
        first_line = f.readline().strip().split()
    file_id = first_line[1] if len(first_line) > 1 else "unknown"
    with open(out_path, "w") as f:
        for start, end in merged:
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} "
                    f"<NA> <NA> speech <NA> <NA>\n")
    log.info(f"    [RTTM] flattened {len(segs)} → {len(merged)} segments → {out_path}")


def _run_meeteval_der(ref_rttm: str, hyp_rttm: str, collar: float = 0.25) -> dict:
    """
    Run meeteval-der md_eval_22 and return the per-reco result dict.
    Returns e.g. {'error_rate': 0.02, 'scored_speaker_time': 250.5,
                  'missed_speaker_time': 2.67, 'falarm_speaker_time': 2.25,
                  'speaker_error_time': 0.14}
    """
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix="_per_reco.json", delete=False) as pf, \
         tempfile.NamedTemporaryFile(suffix="_avg.json", delete=False) as af:
        per_reco_path, avg_path = pf.name, af.name

    cmd = [
        "meeteval-der", "md_eval_22",
        "-r", ref_rttm, "-h", hyp_rttm,
        "--collar", str(collar),
        "--per-reco-out", per_reco_path,
        "--average-out", avg_path,
    ]
    log.info(f"    [DER] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error(f"    [DER] meeteval failed: {proc.stderr}")
        return {}
    for line in proc.stderr.splitlines():
        if line.strip():
            log.info(f"    [DER] {line.strip()}")

    with open(per_reco_path) as f:
        per_reco = json.load(f)
    os.remove(per_reco_path)
    os.remove(avg_path)

    if len(per_reco) == 1:
        return next(iter(per_reco.values()))
    return per_reco


def _eval_der_for_variant(
    label: str,
    hyp_rttm: str,
    ref_rttm: str,
    eval_dir: str,
    session_id: str,
    collar: float,
) -> dict | None:
    """Flatten both to single speaker and run meeteval DER."""
    ref_flat = f"{eval_dir}/{session_id}_{label}_ref.rttm"
    hyp_flat = f"{eval_dir}/{session_id}_{label}_hyp.rttm"
    _flatten_rttm_to_single_speaker(ref_rttm, ref_flat)
    _flatten_rttm_to_single_speaker(hyp_rttm, hyp_flat)
    res = _run_meeteval_der(ref_flat, hyp_flat, collar)
    if res:
        log.info(
            f"    [DER] {label}: DER={res.get('error_rate', 0):.4f} "
            f"(miss={res.get('missed_speaker_time', 0):.1f}s, "
            f"fa={res.get('falarm_speaker_time', 0):.1f}s, "
            f"scored={res.get('scored_speaker_time', 0):.1f}s)"
        )
    return res


def check_der(
    session_id: str,
    mixed_wav: str | None,
    channels: dict[str, str],
    rttms: dict[str, str],
    rttm_out_dir: str | None = None,
    collar: float = 0.25,
) -> DERResult:
    """Legacy single-session DER (unused in batch mode, kept for reference)."""
    raise NotImplementedError("Use batch_der() instead")


# ═══════════════════════════════════════════════════════════════════════════
# Check 3 — AEC-based channel bleed
# ═══════════════════════════════════════════════════════════════════════════
_aec_processor: AEC2ch | None = None


def _get_aec() -> AEC2ch:
    global _aec_processor
    if _aec_processor is None:
        _aec_processor = AEC2ch(enable_aec=True, enable_ns=True, enable_agc=False, stream_delay=0)
    return _aec_processor


def check_channel_bleed(
    channels: dict[str, str],
    rttms: dict[str, str],
    sample_rate: int,
) -> ChannelBleedResult:
    """
    Run AEC per channel (opposite channel as far-end reference).
    Measure echo reduction = baseline silence RMS - post-AEC silence RMS.
    """
    result = ChannelBleedResult()
    if len(channels) < 2:
        return result

    spk_list = list(channels.keys())
    wavs: dict[str, tuple[np.ndarray, int]] = {}
    for spk, path in channels.items():
        log.info(f"    [AEC] reading {spk}: {path}")
        t0 = time.time()
        w = _load_mono_wav_at_declared_sr(path, sample_rate)
        sr = sample_rate
        log.info(f"    [AEC] {spk} loaded ({len(w)/sr:.1f}s, sr={sr}) in {time.time()-t0:.1f}s")
        wavs[spk] = (w, sr)

    sr = sample_rate

    sil_masks: dict[str, np.ndarray] = {}
    for spk, (w, _) in wavs.items():
        rttm_path = rttms.get(spk)
        if rttm_path:
            segs = parse_rttm(rttm_path)
            sil_masks[spk] = silence_mask(segs, len(w), sr, guard_sec=0.05)
        else:
            sil_masks[spk] = np.ones(len(w), dtype=bool)

    aec = _get_aec()
    for i, spk in enumerate(spk_list):
        other_spk = spk_list[1 - i] if len(spk_list) == 2 else spk_list[(i + 1) % len(spk_list)]
        w_near, _ = wavs[spk]
        w_far, _ = wavs[other_spk]

        bl = rms_power_db(w_near, sil_masks[spk])

        log.info(f"    [AEC] running AEC on {spk} (ref={other_spk}) …")
        t0 = time.time()
        w_aec = aec._run_aec(w_near, w_far, sr, desc=f"AEC {spk}")
        log.info(f"    [AEC] {spk} AEC done in {time.time()-t0:.1f}s")

        mask = sil_masks[spk][:len(w_aec)]
        post = rms_power_db(w_aec, mask)
        reduction = bl - post
        result.per_channel_echo_reduction_db[spk] = float(reduction)

        log.info(
            f"    [AEC] {spk}: baseline={bl:.1f} dB → post={post:.1f} dB  "
            f"(reduction={reduction:+.1f} dB)"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Check 4 — SNR (silence power + RTTM-based SNR)
# ═══════════════════════════════════════════════════════════════════════════
def check_snr(
    channels: dict[str, str],
    rttms: dict[str, str],
    sample_rate: int,
) -> SNRResult:
    """
    Per channel:
      - sil = RMS power (dB) averaged over RTTM silence regions
      - SNR = 10*log10(speech_power / noise_power) using RTTM labels
    """
    result = SNRResult()

    for spk, path in channels.items():
        log.info(f"    [SNR] computing for {spk} …")
        t0 = time.time()
        wav = _load_mono_wav_at_declared_sr(path, sample_rate)
        sr = sample_rate

        rttm_path = rttms.get(spk)
        if rttm_path is None:
            log.warning(f"    [SNR] no RTTM for {spk}, skipping")
            continue

        speech_segs = parse_rttm(rttm_path)
        speech_mask = segments_to_mask(speech_segs, len(wav) / sr, sr)
        n = min(len(speech_mask), len(wav))
        speech_mask = speech_mask[:n]
        wav_f64 = wav[:n].astype(np.float64)

        # Silence power (RMS dB averaged over non-speech samples)
        noise_samples = wav_f64[~speech_mask]
        if len(noise_samples) > 0:
            sil_rms = np.sqrt(np.mean(noise_samples ** 2))
            sil_db = float(20.0 * np.log10(max(sil_rms, 1e-10)))
        else:
            sil_db = float("-inf")
        result.per_channel_silence_rms_db[spk] = sil_db

        # SNR
        speech_samples = wav_f64[speech_mask]
        if len(speech_samples) < 100 or len(noise_samples) < 100:
            snr = float("nan")
        else:
            sig_power = np.mean(speech_samples ** 2)
            noi_power = np.mean(noise_samples ** 2)
            snr = 100.0 if noi_power < 1e-20 else float(10.0 * np.log10(sig_power / noi_power))
        result.per_channel_snr_db[spk] = snr

        log.info(f"    [SNR] {spk}: sil={sil_db:.1f} dB, SNR={snr:.1f} dB  ({time.time()-t0:.1f}s)")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Pass / Fail evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_pass_fail(res: SessionQAResult) -> None:
    """Check all metrics against thresholds and set res.passed / res.fail_reasons."""
    fails: list[str] = []

    def _check_max(values: dict[str, float], threshold: float, metric: str):
        for ch, val in values.items():
            if val is not None and val == val and val > threshold:
                fails.append(f"{metric}({ch})={val:.2f}")

    def _check_min(values: dict[str, float], threshold: float, metric: str):
        for ch, val in values.items():
            if val is not None and val == val and val < threshold:
                fails.append(f"{metric}({ch})={val:.1f}")

    # WER: all channels must be ≤ WER_MAX
    _check_max(res.wer.per_channel_wer, WER_MAX, "WER")

    # DER mix: must be ≤ DER_MIX_MAX
    if res.der.mixed_der is not None:
        _check_max({"mix": res.der.mixed_der}, DER_MIX_MAX, "DER")

    # DER per-channel: must be ≤ DER_CH_MAX
    _check_max(res.der.per_channel_der, DER_CH_MAX, "DER")

    # SNR: all channels must be ≥ SNR_MIN_DB
    _check_min(res.snr.per_channel_snr_db, SNR_MIN_DB, "SNR")

    # Echo reduction: all channels must be ≤ ECHO_RED_MAX_DB
    _check_max(res.channel_bleed.per_channel_echo_reduction_db, ECHO_RED_MAX_DB, "red")

    # Silence RMS: all channels must be ≤ SILENCE_RMS_MAX_DB
    _check_max(res.snr.per_channel_silence_rms_db, SILENCE_RMS_MAX_DB, "sil")

    res.fail_reasons = fails
    res.passed = len(fails) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Batch stage runners
# ═══════════════════════════════════════════════════════════════════════════

def batch_wer(manifest: list[dict], results: dict[str, SessionQAResult]):
    """Stage 1: batch ASR on all channel audio files, then eval WER per channel."""
    import tempfile, shutil, subprocess

    # Collect (session_id, ch, audio_path, ref_seglst_path)
    jobs: list[tuple[str, str, str, str]] = []
    for entry in manifest:
        sid = entry["session_id"]
        seglsts = entry.get("seglsts", {})
        for ch, audio_path in entry.get("channels", {}).items():
            ref_path = seglsts.get(ch)
            if ref_path and os.path.isfile(ref_path):
                jobs.append((sid, ch, audio_path, ref_path))
            elif ref_path:
                log.warning(f"[Stage 1] WER: missing seglst file {ref_path!r} for {sid} {ch}")
            else:
                log.warning(f"[Stage 1] WER: no seglsts['{ch}'] for session {sid}")

    if not jobs:
        log.warning("[Stage 1] WER: no jobs to run")
        return

    audio_paths = [j[2] for j in jobs]  # audio_path
    n_files = len(audio_paths)
    # batch_size=1 for long-form audio to avoid OOM from padding;
    # NeMo still processes all files in a single transcribe() call
    log.info(f"[Stage 1] WER: transcribing {n_files} files with Parakeet …")

    model = _get_parakeet_model()
    t0 = time.time()
    asr_outputs = model.transcribe(audio_paths, batch_size=1)
    log.info(f"[Stage 1] WER: batch transcription done in {time.time()-t0:.1f}s")

    eval_dir = tempfile.mkdtemp(prefix="qa_wer_batch_")
    all_ref_segs: list[dict] = []
    all_hyp_segs: list[dict] = []

    for (sid, ch, _, ref_ch_path), asr_out in zip(jobs, asr_outputs):
        raw_hyp = asr_out.text.strip()
        hyp_text = _normalize_hyp(raw_hyp)
        sid_ch = f"{sid}_{ch}"

        with open(ref_ch_path) as f:
            ref_segs = json.load(f)
        spk = ref_segs[0]["speaker"] if ref_segs else ch
        ref_merged = " ".join(seg["words"] for seg in ref_segs if seg.get("words", "").strip())

        all_ref_segs.append({"session_id": sid_ch, "words": ref_merged,
                             "start_time": "0.00", "end_time": "9999.99", "speaker": spk})
        all_hyp_segs.append({"session_id": sid_ch, "words": hyp_text,
                             "start_time": "0.00", "end_time": "9999.99", "speaker": spk})

    ref_path = f"{eval_dir}/all_ref.seglst.json"
    hyp_path = f"{eval_dir}/all_hyp.seglst.json"
    with open(ref_path, "w") as f:
        json.dump(all_ref_segs, f)
    with open(hyp_path, "w") as f:
        json.dump(all_hyp_segs, f)

    wer_results = _run_meeteval_wer(ref_path, hyp_path)

    for (sid, ch, _, _), ref_seg in zip(jobs, all_ref_segs):
        sid_ch = ref_seg["session_id"]
        if isinstance(wer_results, dict) and sid_ch in wer_results:
            er = wer_results[sid_ch].get("error_rate")
        elif isinstance(wer_results, dict) and "error_rate" in wer_results:
            er = wer_results.get("error_rate")
        else:
            er = None
        if er is not None:
            results[sid].wer.per_channel_wer[ch] = float(er)
            log.info(f"  [WER] {sid} {ch}: WER={er:.4f}")

    shutil.rmtree(eval_dir, ignore_errors=True)


def _batch_sortformer(audio_paths: list[str], batch_size: int = 8) -> list[list[str]]:
    """Batch Sortformer diarization on multiple files at once."""
    model = _get_sortformer_model()
    t0 = time.time()
    log.info(f"    [Sortformer] batch diarize {len(audio_paths)} files (batch_size={batch_size}) …")
    all_preds = model.diarize(audio=audio_paths, batch_size=batch_size)
    log.info(f"    [Sortformer] batch done in {time.time()-t0:.1f}s")
    return all_preds


def _batch_silero(
    audio_paths: list[str],
    declared_sample_rates: list[int],
) -> list[list[str]]:
    """Run Silero VAD on each file sequentially (no native batch API)."""
    model = _get_silero_model()
    get_speech_timestamps = model._utils[0]

    all_results: list[list[str]] = []
    t0 = time.time()
    for audio_path, declared_sr in zip(audio_paths, declared_sample_rates):
        wav_np = _load_mono_wav_at_declared_sr(audio_path, declared_sr)
        wav, silero_sr = _wav_tensor_for_silero_vad(wav_np, declared_sr)
        timestamps = get_speech_timestamps(
            wav, model, sampling_rate=silero_sr, return_seconds=True
        )
        segs = [f"{ts['start']:.3f} {ts['end']:.3f} speech" for ts in timestamps]
        all_results.append(segs)
    log.info(f"    [Silero] sequential VAD on {len(audio_paths)} files in {time.time()-t0:.1f}s")
    return all_results


def batch_der(manifest: list[dict], results: dict[str, SessionQAResult],
              rttm_out_dir: str | None = None, collar: float = 0.25):
    """Stage 2: run SAD on all audio (mixed+channels), then eval DER."""
    import tempfile, shutil

    # Collect all SAD jobs: (sid, label, audio_path, sample_rate)
    sad_jobs: list[tuple[str, str, str, int]] = []
    for entry in manifest:
        sid = entry["session_id"]
        channels = entry.get("channels", {})
        rttms = entry.get("rttms", {})
        if not channels or not rttms:
            continue
        sr = _parse_sample_rate(entry)
        mixed_wav = entry.get("mixed_wav") or next(iter(channels.values()), None)
        if mixed_wav:
            sad_jobs.append((sid, "mixed", mixed_wav, sr))
        for ch, ch_path in channels.items():
            sad_jobs.append((sid, ch, ch_path, sr))

    if not sad_jobs:
        log.warning("[Stage 2] DER: no jobs to run")
        return

    audio_paths = [j[2] for j in sad_jobs]
    silero_srs = [j[3] for j in sad_jobs]
    n_files = len(audio_paths)
    log.info(f"[Stage 2] DER: running SAD on {n_files} files (Sortformer batch + Silero) …")

    out_dir = rttm_out_dir or tempfile.mkdtemp(prefix="qa_rttm_")
    os.makedirs(out_dir, exist_ok=True)
    eval_dir = tempfile.mkdtemp(prefix="qa_eval_batch_")

    # Batch Sortformer
    all_sf = _batch_sortformer(audio_paths, batch_size=8)
    # Batch Silero (sequential internally)
    all_si = _batch_silero(audio_paths, silero_srs)

    # Union Sortformer + Silero per file, write hypothesis RTTMs
    hyp_rttm_paths: dict[tuple[str, str], str] = {}
    for (sid, label, _, _), sf_segs, si_segs in zip(sad_jobs, all_sf, all_si):
        sf_merged = merge_segments(sorted(_segs_str_to_tuples(sf_segs)))
        si_merged = merge_segments(sorted(_segs_str_to_tuples(si_segs)))
        unioned = merge_segments(sf_merged + si_merged)
        union_strs = [f"{s:.3f} {e:.3f} speech" for s, e in unioned]

        rttm_path = f"{out_dir}/{sid}_{label}.rttm"
        _write_rttm(union_strs, rttm_path, sid)
        hyp_rttm_paths[(sid, label)] = rttm_path
        log.info(f"  [SAD] {sid}/{label}: Sortformer={len(sf_merged)}, "
                 f"Silero={len(si_merged)} → union={len(unioned)} segs")

    # Evaluate DER for each session
    log.info(f"[Stage 2] DER: evaluating {len(sad_jobs)} DER values …")
    for entry in manifest:
        sid = entry["session_id"]
        channels = entry.get("channels", {})
        rttms = entry.get("rttms", {})
        if not channels or not rttms:
            continue

        result = results[sid].der

        # Mixed DER
        hyp_mixed = hyp_rttm_paths.get((sid, "mixed"))
        if hyp_mixed:
            ref_merged_raw = f"{eval_dir}/{sid}_ref_merged_raw.rttm"
            with open(ref_merged_raw, "w") as fout:
                for rttm_path in rttms.values():
                    with open(rttm_path) as fin:
                        fout.write(fin.read())
            res = _eval_der_for_variant("mixed", hyp_mixed, ref_merged_raw,
                                        eval_dir, sid, collar)
            if res:
                result.mixed_der = res.get("error_rate")
                result.detail["mixed"] = res

        # Per-channel DER
        for ch in channels:
            ref_path = rttms.get(ch)
            hyp_path = hyp_rttm_paths.get((sid, ch))
            if ref_path and hyp_path:
                res = _eval_der_for_variant(ch, hyp_path, ref_path, eval_dir, sid, collar)
                if res:
                    result.per_channel_der[ch] = res.get("error_rate", float("nan"))
                    result.detail[ch] = res
                    log.info(f"  [DER] {sid} {ch}: DER={result.per_channel_der[ch]:.4f}")

    shutil.rmtree(eval_dir, ignore_errors=True)


def batch_aec(manifest: list[dict], results: dict[str, SessionQAResult]):
    """Stage 3: AEC-based channel bleed for all sessions."""
    n = sum(1 for e in manifest if len(e.get("channels", {})) >= 2)
    log.info(f"[Stage 3] AEC: processing {n} sessions …")

    for entry in manifest:
        sid = entry["session_id"]
        channels = entry.get("channels", {})
        rttms = entry.get("rttms", {})
        if len(channels) < 2:
            continue
        log.info(f"  [AEC] {sid} …")
        sr = _parse_sample_rate(entry)
        results[sid].channel_bleed = check_channel_bleed(channels, rttms, sr)
        for ch, red in results[sid].channel_bleed.per_channel_echo_reduction_db.items():
            log.info(f"    {ch}: reduction={red:+.1f} dB")


def batch_snr(manifest: list[dict], results: dict[str, SessionQAResult]):
    """Stage 4: SNR for all sessions."""
    n = sum(1 for e in manifest if e.get("channels"))
    log.info(f"[Stage 4] SNR: processing {n} sessions …")

    for entry in manifest:
        sid = entry["session_id"]
        channels = entry.get("channels", {})
        rttms = entry.get("rttms", {})
        if not channels:
            continue
        log.info(f"  [SNR] {sid} …")
        sr = _parse_sample_rate(entry)
        results[sid].snr = check_snr(channels, rttms, sr)


def _fmt(val, fmt_str=".4f"):
    if val is None or (isinstance(val, float) and val != val):
        return "N/A"
    return f"{val:{fmt_str}}"


def print_summary(results: list[SessionQAResult]):
    hdr = (
        f"{'Session':>40s}  "
        f"{'WER_1':>7s}  {'WER_2':>7s}  "
        f"{'DERmix':>7s}  {'DER_1':>7s}  {'DER_2':>7s}  "
        f"{'red_1':>7s}  {'red_2':>7s}  "
        f"{'sil_1':>7s}  {'sil_2':>7s}  "
        f"{'SNR_1':>7s}  {'SNR_2':>7s}  "
        f"{'Result'}"
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        wer_vals = list(r.wer.per_channel_wer.values())
        der_vals = list(r.der.per_channel_der.values())
        red_vals = list(r.channel_bleed.per_channel_echo_reduction_db.values())
        sil_vals = list(r.snr.per_channel_silence_rms_db.values())
        snr_vals = list(r.snr.per_channel_snr_db.values())

        if r.passed:
            verdict = "PASS"
        else:
            verdict = "FAIL  ← " + ", ".join(r.fail_reasons)

        print(
            f"{r.session_id:>40s}  "
            f"{_fmt(wer_vals[0] if len(wer_vals) > 0 else None):>7s}  "
            f"{_fmt(wer_vals[1] if len(wer_vals) > 1 else None):>7s}  "
            f"{_fmt(r.der.mixed_der):>7s}  "
            f"{_fmt(der_vals[0] if len(der_vals) > 0 else None):>7s}  "
            f"{_fmt(der_vals[1] if len(der_vals) > 1 else None):>7s}  "
            f"{_fmt(red_vals[0] if len(red_vals) > 0 else None, '+.1f'):>7s}  "
            f"{_fmt(red_vals[1] if len(red_vals) > 1 else None, '+.1f'):>7s}  "
            f"{_fmt(sil_vals[0] if len(sil_vals) > 0 else None, '.1f'):>7s}  "
            f"{_fmt(sil_vals[1] if len(sil_vals) > 1 else None, '.1f'):>7s}  "
            f"{_fmt(snr_vals[0] if len(snr_vals) > 0 else None, '.1f'):>7s}  "
            f"{_fmt(snr_vals[1] if len(snr_vals) > 1 else None, '.1f'):>7s}  "
            f"{verdict}"
        )
    print("=" * len(hdr))

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(f"\n  Summary: {n_pass} PASS / {n_fail} FAIL  (out of {len(results)} sessions)\n")
    print(f"  Thresholds (session FAILs if ANY channel exceeds ANY threshold):")
    print(f"    WER    ≤ {WER_MAX:.0%}  (Parakeet-TDT-0.6B-v3, per-channel, meeteval wer, chime8 norm)")
    print(f"    DERmix ≤ {DER_MIX_MAX:.0%}  (mixed-signal SAD error, md_eval_22, collar=0.25)")
    print(f"    DERch  ≤ {DER_CH_MAX:.0%}  (per-channel SAD error, Sortformer∪Silero)")
    print(f"    SNR  ≥ {SNR_MIN_DB:.0f} dB  (RTTM-based signal-to-noise ratio)")
    print(f"    red  ≤ {ECHO_RED_MAX_DB:.0f} dB  (AEC echo reduction; high = heavy channel bleed)")
    print(f"    sil  ≤ {SILENCE_RMS_MAX_DB:.0f} dB (silence floor RMS power)")


def main():
    parser = argparse.ArgumentParser(description="Run QA checks on audio sessions")
    parser.add_argument("--file_list", default="file_list.json")
    parser.add_argument("--output", default="qa_results.json")
    parser.add_argument("--rttm_out_dir", default=None,
                        help="Directory to write hypothesis RTTM outputs")
    parser.add_argument("--check_aec", action="store_true",
                        help="Run optional AEC channel bleed check (Stage 3); skipped by default")
    args = parser.parse_args()

    with open(args.file_list) as f:
        manifest = json.load(f)
    log.info(f"Loaded {len(manifest)} sessions from {args.file_list}")
    _validate_manifest_sample_rates(manifest)
    sys.stdout.flush()

    results: dict[str, SessionQAResult] = {
        e["session_id"]: SessionQAResult(session_id=e["session_id"])
        for e in manifest
    }

    t_total = time.time()

    t0 = time.time()
    batch_wer(manifest, results)
    log.info(f"[Stage 1] WER stage completed in {time.time()-t0:.1f}s\n")
    sys.stdout.flush()

    t0 = time.time()
    batch_der(manifest, results, rttm_out_dir=args.rttm_out_dir)
    log.info(f"[Stage 2] DER stage completed in {time.time()-t0:.1f}s\n")
    sys.stdout.flush()

    if args.check_aec:
        t0 = time.time()
        batch_aec(manifest, results)
        log.info(f"[Stage 3] AEC stage completed in {time.time()-t0:.1f}s\n")
    else:
        log.info("[Stage 3] AEC: SKIPPED (pass --check_aec to enable)\n")
    sys.stdout.flush()

    t0 = time.time()
    batch_snr(manifest, results)
    log.info(f"[Stage 4] SNR stage completed in {time.time()-t0:.1f}s\n")
    sys.stdout.flush()

    for res in results.values():
        evaluate_pass_fail(res)

    result_list = [results[e["session_id"]] for e in manifest]
    print_summary(result_list)
    log.info(f"Total pipeline time: {time.time()-t_total:.1f}s")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in result_list], f, indent=2, cls=NumpyEncoder)
    log.info(f"Detailed results → {args.output}")


if __name__ == "__main__":
    main()
