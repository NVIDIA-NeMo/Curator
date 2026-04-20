#!/usr/bin/env python3
"""Score UTMOSv2 MOS for shards by streaming audio from AIS.

GPU-accelerated spectrograms via torchaudio (35x faster than default librosa).

Usage::

    python score_utmos_shard.py \
        --manifest_dir /path/to/manifests \
        --tar_pattern "s3://YTC/ru/webds_ru/audio_{SHARD}.tar" \
        --output_dir /output/utmos \
        --shard_ids 0
"""
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
             "TORCH_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import io
import json
import sys
import tarfile
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

sys.stdout.reconfigure(line_buffering=True)

_DEFAULT_TOKEN = os.environ.get("AIS_AUTHN_TOKEN", "")


# ---------------------------------------------------------------------------
# GPU spectrogram acceleration for UTMOSv2
# ---------------------------------------------------------------------------

def _patch_utmosv2_for_gpu():
    """Monkey-patch UTMOSv2 to compute spectrograms on GPU instead of CPU.

    Replaces the dataset to return raw audio segments and wraps the model
    to compute mel spectrograms on GPU via torchaudio.  ~35x faster.
    """
    from utmosv2.dataset._base import BaseDataset, DataDomainMixin
    from utmosv2.dataset._utils import extend_audio, select_random_start
    from utmosv2.dataset import SSLExtDataset
    import utmosv2.utils as utils
    import utmosv2._core.model._common as common

    class _GPURawAudioDataset(BaseDataset):
        """Returns raw audio segments for GPU spec computation."""
        def __init__(self, cfg, data, phase, transform=None):
            super().__init__(cfg, data, phase, transform)
            self.ssl = SSLExtDataset(cfg, data, phase)
            self.spec_audio_len = int(cfg.dataset.spec_frames.frame_sec * cfg.sr)
            self.num_specs = len(cfg.dataset.specs)
            self.num_frames = cfg.dataset.spec_frames.num_frames
            self.mixup_inner = cfg.dataset.spec_frames.mixup_inner
            self.mixup_alpha = cfg.dataset.spec_frames.mixup_alpha

        def __getitem__(self, idx):
            x1, d, target = self.ssl[idx]
            y, _ = self._get_audio_and_mos(idx)
            y = extend_audio(y, self.spec_audio_len, method="tile")
            segments, lambdas = [], []
            for _ in range(self.num_frames):
                y1 = select_random_start(y, self.spec_audio_len)
                for _ in range(self.num_specs):
                    segments.append(torch.from_numpy(y1.copy()).float())
                    if self.mixup_inner:
                        y2 = select_random_start(y, self.spec_audio_len)
                        segments.append(torch.from_numpy(y2.copy()).float())
                        lambdas.append(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                    else:
                        segments.append(torch.zeros(self.spec_audio_len))
                        lambdas.append(1.0)
            return (x1, torch.stack(segments),
                    torch.tensor(lambdas, dtype=torch.float32), d, target)

    # Patch dataset factory
    _orig_get_dataset = utils.get_dataset
    def _patched_get_dataset(cfg, data, phase, transform=None):
        if cfg.dataset.name == "ssl_multispec_ext":
            return _GPURawAudioDataset(cfg, data, phase, transform)
        return _orig_get_dataset(cfg, data, phase, transform)
    utils.get_dataset = _patched_get_dataset
    common.get_dataset = _patched_get_dataset

    # Patch predict loop for 5-tuple (x1, segments, lambdas, d, target)
    def _patched_predict_impl(self, dataloader, num_repetitions, device, verbose):
        from tqdm import tqdm
        self._model.eval().to(device)
        res = 0.0
        for _rep in range(num_repetitions):
            pred = []
            pbar = tqdm(dataloader, disable=not verbose, total=len(dataloader),
                        desc="Predicting: ")
            with torch.no_grad():
                for t in pbar:
                    x = [v.to(device, non_blocking=True) for v in t[:-1]]
                    with torch.amp.autocast("cuda"):
                        output = self._model(*x).squeeze(1)
                    pred.append(output.cpu().numpy())
            res += np.concatenate(pred) / num_repetitions
        return res
    common.UTMOSv2ModelMixin._predict_impl = _patched_predict_impl


class _GPUSpecModel(nn.Module):
    """Wraps UTMOSv2 model to compute spectrograms on GPU via torchaudio."""

    def __init__(self, original_model: nn.Module, cfg):
        super().__init__()
        self.cfg = cfg
        self.ssl = original_model.ssl
        self.spec_long = original_model.spec_long
        self.fc = original_model.fc
        self.num_dataset = getattr(original_model, "num_dataset", 10)
        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sr, n_fft=s.n_fft, hop_length=s.hop_length,
                win_length=s.win_length, n_mels=s.n_mels, power=2.0,
                pad_mode="constant", norm="slaney", mel_scale="slaney",
            ) for s in cfg.dataset.specs
        ])
        self.spec_norms = [s.norm for s in cfg.dataset.specs]

    def _audio_to_spec(self, audio: torch.Tensor, spec_idx: int) -> torch.Tensor:
        """(batch, audio_len) -> (batch, 3, 512, 512) mel spectrogram."""
        spec = self.mel_transforms[spec_idx](audio)
        log_spec = 10.0 * torch.log10(spec.clamp(min=1e-10))
        ref_vals = 10.0 * torch.log10(
            spec.reshape(spec.shape[0], -1).max(dim=1).values.clamp(min=1e-10))
        log_spec = torch.clamp(log_spec - ref_vals[:, None, None], min=-80.0)
        norm = self.spec_norms[spec_idx]
        if norm is not None:
            log_spec = (log_spec + norm) / norm
        log_spec = log_spec.unsqueeze(1).expand(-1, 3, -1, -1)
        return torch.nn.functional.interpolate(
            log_spec, size=(512, 512), mode="bilinear", align_corners=False)

    def forward(self, x1, segments, lambdas, d):
        B = x1.shape[0]
        zero_d = torch.zeros(B, self.num_dataset, device=x1.device)
        ssl_out = self.ssl(x1, zero_d)

        specs = []
        si, li = 0, 0
        for _ in range(self.cfg.dataset.spec_frames.num_frames):
            for spi in range(len(self.cfg.dataset.specs)):
                s1 = self._audio_to_spec(segments[:, si], spi)
                s2 = self._audio_to_spec(segments[:, si + 1], spi)
                si += 2
                lmd = lambdas[:, li].reshape(-1, 1, 1, 1)
                li += 1
                specs.append(lmd * s1 + (1 - lmd) * s2)

        x2 = torch.stack(specs, dim=1)
        spec_out = self.spec_long(x2, zero_d)
        return self.fc(torch.cat([ssl_out, spec_out, d], dim=1))


# ---------------------------------------------------------------------------
# AIS / tar helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UTMOSv2 scoring via AIS streaming")
    p.add_argument("--manifest_dir", required=True)
    p.add_argument("--tar_pattern", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_ids", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--ais_token", default="")
    return p.parse_args()


def init_ais_client(token: str = ""):
    from aistore.sdk import Client
    endpoint = os.environ.get("AIS_ENDPOINT", "http://localhost:51080")
    token = token or _DEFAULT_TOKEN
    try:
        return Client(endpoint, token=token)
    except TypeError:
        client = Client(endpoint)
        os.environ["AIS_AUTHN_TOKEN"] = token
        return client


def download_tar(client, tar_url: str, local_path: str) -> None:
    from urllib.parse import urlparse
    parsed = urlparse(tar_url)
    obj = client.bucket(parsed.netloc, provider="aws").object(parsed.path.lstrip("/"))
    reader = getattr(obj, "get_reader", None)
    raw = reader().read_all() if reader else obj.get().read_all()
    with open(local_path, "wb") as f:
        f.write(raw)


def load_tar_members(tar_path: str) -> dict[str, bytes]:
    members = {}
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if m.isfile():
                stream = tf.extractfile(m)
                if stream is not None:
                    data = stream.read()
                    members[m.name] = data
                    bn = os.path.basename(m.name)
                    if bn not in members:
                        members[bn] = data
    return members


def audio_bytes_to_wav(audio_bytes: bytes, target_sr: int) -> bytes | None:
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio[:, 0]
    if len(audio) == 0:
        return None
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            ratio = target_sr / sr
            indices = np.round(np.arange(0, len(audio), 1 / ratio)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Per-shard processing
# ---------------------------------------------------------------------------

def process_shard(shard_id: int, gpu_id: int, args: argparse.Namespace) -> None:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "," in cvd or cvd == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    gpu_label = os.environ["CUDA_VISIBLE_DEVICES"]

    manifest_path = os.path.join(args.manifest_dir, f"shard_{shard_id}.jsonl")
    out_path = os.path.join(args.output_dir, f"shard_{shard_id}.jsonl")

    if not os.path.exists(manifest_path):
        print(f"[GPU {gpu_label}] Shard {shard_id}: manifest not found, skipping")
        return
    if os.path.exists(out_path):
        print(f"[GPU {gpu_label}] Shard {shard_id}: already done, skipping")
        return

    t0 = time.time()
    with open(manifest_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"[GPU {gpu_label}] Shard {shard_id}: {len(rows)} rows")

    # Download tar
    tar_url = args.tar_pattern.replace("{SHARD}", str(shard_id))
    client = init_ais_client(args.ais_token)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_tar = tmp.name
    try:
        download_tar(client, tar_url, tmp_tar)
        size_mb = os.path.getsize(tmp_tar) / (1024 * 1024)
        print(f"[GPU {gpu_label}] Shard {shard_id}: downloaded {size_mb:.0f} MB in {time.time()-t0:.1f}s")
        members = load_tar_members(tmp_tar)
    finally:
        os.unlink(tmp_tar)

    # Patch UTMOSv2 for GPU spectrograms and load model
    _patch_utmosv2_for_gpu()
    import utmosv2
    model = utmosv2.create_model(pretrained=True)
    model._model = _GPUSpecModel(model._model, model._cfg)
    print(f"[GPU {gpu_label}] Shard {shard_id}: model loaded (GPU specs) in {time.time()-t0:.1f}s")

    # Convert + score
    scored = 0
    failed = 0
    with tempfile.TemporaryDirectory() as wav_dir:
        valid_indices = []
        wav_count = 0
        for i, row in enumerate(rows):
            afp = row.get("audio_filepath", "")
            audio_bytes = members.get(afp) or members.get(os.path.basename(afp))
            if audio_bytes is None:
                row["utmosv2_score"] = float("nan")
                failed += 1
                continue
            wav_bytes = audio_bytes_to_wav(audio_bytes, args.target_sr)
            if wav_bytes is None:
                row["utmosv2_score"] = float("nan")
                failed += 1
                continue
            wav_path = os.path.join(wav_dir, f"{wav_count:08d}.wav")
            with open(wav_path, "wb") as wf:
                wf.write(wav_bytes)
            valid_indices.append(i)
            wav_count += 1

        print(f"[GPU {gpu_label}] Shard {shard_id}: {wav_count} WAVs ready, {failed} failed, scoring...")
        if valid_indices:
            t2 = time.time()
            results = model.predict(
                input_dir=wav_dir, batch_size=args.batch_size,
                num_repetitions=1, predict_dataset="sarulab",
                num_workers=0, verbose=False,
            )
            scores = [r["predicted_mos"] for r in results]
            for idx, score in zip(valid_indices, scores):
                rows[idx]["utmosv2_score"] = round(float(score), 4)
                scored += 1
            print(f"[GPU {gpu_label}] Shard {shard_id}: scored {scored} in {time.time()-t2:.1f}s")

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"[GPU {gpu_label}] Shard {shard_id}: done ({scored} scored, {failed} failed, {elapsed:.1f}s)")


def main() -> None:
    args = parse_args()
    shard_ids = [int(s) for s in args.shard_ids.split(",") if s.strip()]
    for sid in shard_ids:
        process_shard(sid, 0, args)


if __name__ == "__main__":
    main()
