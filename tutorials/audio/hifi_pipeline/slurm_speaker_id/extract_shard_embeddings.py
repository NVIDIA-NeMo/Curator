#!/usr/bin/env python3
"""Extract speaker embeddings for a single shard.

Reads a NeMo JSONL manifest from Lustre, downloads the matching tar from S3,
extracts audio members, runs TitaNet inference, and saves embeddings_N.npz.

Usage::

    python extract_shard_embeddings.py \
        --manifest_path /path/to/shard_42.jsonl \
        --tar_url s3://yodas2/ru/0_from_captions/audio_42.tar \
        --output_dir /output/embeddings \
        --shard_id 42
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import tempfile
import time

import numpy as np
import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_path", required=True, help="Path to shard_N.jsonl on Lustre")
    p.add_argument("--tar_url", required=True, help="S3 URL of audio_N.tar")
    p.add_argument("--output_dir", required=True, help="Output directory for embeddings_N.npz")
    p.add_argument("--shard_id", type=int, required=True)
    p.add_argument("--model_name", default="nvidia/speakerverification_en_titanet_large")
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for TitaNet inference")
    p.add_argument("--s3cfg", default=os.path.expanduser("~/.s3cfg"), help="Path to s3cfg")
    p.add_argument("--s3_endpoint", default="", help="Override S3 endpoint URL")
    p.add_argument("--ais_token", default="", help="AIS auth token (overrides AIS_AUTHN_TOKEN env)")
    p.add_argument("--skip_filtered", action="store_true", help="Skip rows where filtered_out=True")
    return p.parse_args()


_DEFAULT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVzdGVycyI6bnVsbCwiYWRtaW4iOnRydWUsImlzcyI6Imh0dHBzOi8vbG9jYWxob3N0OjUyMDAxIiwic3ViIjoiYWRtaW4iLCJleHAiOjI0MDY1NzY3ODgsImlhdCI6MTc3NTg1Njc4OH0.NuwKfhdXBaOXYxx4eTataX7XWP1wEOwtopXhFGzppkw"


def init_ais_client(token_override: str = ""):
    """Initialize AIStore client with auth token injected into session."""
    from aistore.sdk import Client

    endpoint = os.environ.get("AIS_ENDPOINT", "http://asr.iad.oci.aistore.nvidia.com:51080")
    token = token_override or _DEFAULT_TOKEN
    client = Client(endpoint)
    if token:
        client._request_client._session.headers["Authorization"] = f"Bearer {token}"
    return client


def download_tar(client, tar_url: str, local_path: str) -> None:
    """Download tar from AIS/S3 to local path."""
    from urllib.parse import urlparse

    parsed = urlparse(tar_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    obj = client.bucket(bucket, provider="aws").object(key)
    # Support both old and new aistore SDK API
    reader = getattr(obj, "get_reader", None)
    if reader:
        raw = reader().read_all()
    else:
        raw = obj.get().read_all()
    with open(local_path, "wb") as f:
        f.write(raw)


def load_tar_members(tar_path: str) -> dict[str, bytes]:
    """Load all audio members from a tar file into memory."""
    members = {}
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if m.isfile():
                stream = tf.extractfile(m)
                if stream is not None:
                    data = stream.read()
                    members[m.name] = data
                    # Also store by basename for flexible matching
                    basename = os.path.basename(m.name)
                    if basename not in members:
                        members[basename] = data
    return members


def audio_to_tensor(audio_bytes: bytes, target_sr: int) -> np.ndarray | None:
    """Decode audio bytes to float32 mono array at target_sr."""
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            # Manual resample fallback (nearest neighbor, low quality)
            ratio = target_sr / sr
            indices = np.round(np.arange(0, len(audio), 1 / ratio)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]
    return audio.astype(np.float32)


@torch.no_grad()
def extract_embeddings_batch(
    model, audios: list[np.ndarray], device: str
) -> np.ndarray:
    """Run TitaNet on a batch of audio arrays, return (N, D) embeddings."""
    max_len = max(a.shape[0] for a in audios)
    batch = np.zeros((len(audios), max_len), dtype=np.float32)
    lengths = np.zeros(len(audios), dtype=np.int64)
    for i, a in enumerate(audios):
        batch[i, :a.shape[0]] = a
        lengths[i] = a.shape[0]

    signals = torch.tensor(batch, device=device, dtype=torch.float32)
    signal_lens = torch.tensor(lengths, device=device)
    _, embs = model.forward(input_signal=signals, input_signal_length=signal_lens)
    return embs.cpu().numpy()


def main():
    args = parse_args()
    t0 = time.time()

    # Read manifest
    with open(args.manifest_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    if args.skip_filtered:
        rows = [r for r in rows if not r.get("filtered_out", False)]

    if not rows:
        print(f"Shard {args.shard_id}: 0 rows after filtering, skipping")
        return

    print(f"Shard {args.shard_id}: {len(rows)} rows from {args.manifest_path}")

    # Download tar from S3
    print(f"Downloading {args.tar_url} ...")
    ais_client = init_ais_client(args.ais_token)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_tar = tmp.name
    try:
        download_tar(ais_client, args.tar_url, tmp_tar)
        tar_size_mb = os.path.getsize(tmp_tar) / (1024 * 1024)
        print(f"Downloaded {tar_size_mb:.1f} MB in {time.time() - t0:.1f}s")

        # Load tar members into memory
        t1 = time.time()
        members = load_tar_members(tmp_tar)
        print(f"Loaded {len(members)} tar members in {time.time() - t1:.1f}s")
    finally:
        os.unlink(tmp_tar)

    # Load model
    import nemo.collections.asr as nemo_asr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name=args.model_name, map_location=device
    )
    model.eval()
    print(f"TitaNet loaded on {device}")

    # Process rows in batches
    cut_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    failed = 0

    batch_audios: list[np.ndarray] = []
    batch_ids: list[str] = []

    for i, row in enumerate(rows):
        afp = row.get("audio_filepath", "")
        audio_bytes = members.get(afp) or members.get(os.path.basename(afp))
        if audio_bytes is None:
            failed += 1
            continue

        audio = audio_to_tensor(audio_bytes, args.target_sr)
        if audio is None or audio.shape[0] == 0:
            failed += 1
            continue

        batch_audios.append(audio)
        batch_ids.append(afp)

        if len(batch_audios) >= args.batch_size:
            embs = extract_embeddings_batch(model, batch_audios, device)
            all_embeddings.append(embs)
            cut_ids.extend(batch_ids)
            batch_audios.clear()
            batch_ids.clear()

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(rows)}] extracted={len(cut_ids)}, failed={failed}")

    # Flush remaining batch
    if batch_audios:
        embs = extract_embeddings_batch(model, batch_audios, device)
        all_embeddings.append(embs)
        cut_ids.extend(batch_ids)

    if not all_embeddings:
        print(f"Shard {args.shard_id}: no embeddings extracted! ({failed} failures)")
        sys.exit(1)

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"embeddings_{args.shard_id}.npz")
    np.savez(
        out_path,
        cut_ids=np.array(cut_ids, dtype=object),
        embeddings=embeddings,
    )
    elapsed = time.time() - t0
    print(
        f"Shard {args.shard_id}: saved {embeddings.shape[0]} embeddings "
        f"(dim={embeddings.shape[1]}) -> {out_path} "
        f"({failed} failed, {elapsed:.1f}s total)"
    )


if __name__ == "__main__":
    main()
