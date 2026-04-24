#!/usr/bin/env python3
"""Run a single pipeline stage on a single shard.

Unified entry point for all stages. Streams audio from AIS/S3 NeMo tars,
runs the specified stage, writes output JSONL.

Usage::

    python run_stage.py \
        --stage sed \
        --manifest_path /path/to/shard_0.jsonl \
        --tar_url s3://YTC/ru/webds_ru/audio_0.tar \
        --output_dir /output/sed \
        --shard_id 0

Stages: sed, sed_post, segment, diarize, transcribe, embed, group_video, cluster, utmos
"""
from __future__ import annotations


import argparse
import io
import json
import os
import tarfile
import tempfile
import sys
import time

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(line_buffering=True)

_DEFAULT_AIS_TOKEN = os.environ.get("AIS_AUTHN_TOKEN", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single HIFI pipeline stage on a shard")
    p.add_argument("--stage", required=True, choices=[
        "sed", "sed_post", "segment", "diarize", "transcribe",
        "embed", "group_video", "cluster", "utmos",
    ])
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--tar_url", default="", help="S3/AIS URL of audio tar (for stages needing audio)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_id", type=int, required=True)

    # Audio options
    p.add_argument("--target_sr", type=int, default=16000)

    # SED options
    p.add_argument("--sed_checkpoint", default="")
    p.add_argument("--sed_model_type", default="Cnn14_DecisionLevelMax")
    p.add_argument("--sed_threshold", type=float, default=0.5)

    # Transcription options
    p.add_argument("--language", default="Ru")
    p.add_argument("--omni_model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--llm_model", default="Qwen/Qwen3-30B-A3B-Instruct")
    p.add_argument("--vllm_host", default="localhost")
    p.add_argument("--vllm_port", type=int, default=8200)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tensor_parallel_size", type=int, default=2)

    # Speaker ID options
    p.add_argument("--speaker_model", default="nvidia/speakerverification_en_titanet_large")
    p.add_argument("--embedding_dir", default="", help="Embedding dir for clustering stage")
    p.add_argument("--cluster_threshold", type=float, default=0.292)

    # UTMOS options
    p.add_argument("--utmos_batch_size", type=int, default=16)

    # General
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


# ---- AIS/tar helpers (shared across stages) ----

def init_ais_client(token: str = ""):
    from aistore.sdk import Client
    endpoint = os.environ.get("AIS_ENDPOINT", "http://localhost:51080")
    token = token or _DEFAULT_AIS_TOKEN
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


def load_tar_audio(tar_path: str, target_sr: int = 16000) -> dict[str, tuple[np.ndarray, int]]:
    """Load all audio from tar into memory as {filename: (waveform, sample_rate)}."""
    members = {}
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            stream = tf.extractfile(m)
            if stream is None:
                continue
            raw = stream.read()
            try:
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                members[m.name] = (audio, sr)
                bn = os.path.basename(m.name)
                if bn not in members:
                    members[bn] = (audio, sr)
            except Exception:
                continue
    return members


def read_manifest(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_manifest(rows: list[dict], output_dir: str, shard_id: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"shard_{shard_id}.jsonl")
    with open(out_path, "w") as f:
        for row in rows:
            # Remove non-serializable fields
            clean = {k: v for k, v in row.items()
                     if not isinstance(v, (np.ndarray, bytes, memoryview))}
            f.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")
    return out_path


def stream_tar_if_needed(args) -> tuple[dict | None, str | None]:
    """Download and load tar if tar_url is provided. Returns (audio_dict, tmp_path)."""
    if not args.tar_url:
        return None, None
    client = init_ais_client()
    tmp = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
    tmp_path = tmp.name
    tmp.close()
    t0 = time.time()
    download_tar(client, args.tar_url, tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    print(f"Downloaded {args.tar_url} ({size_mb:.1f} MB in {time.time()-t0:.1f}s)")
    audio = load_tar_audio(tmp_path, args.target_sr)
    os.unlink(tmp_path)
    print(f"Loaded {len(audio)} audio members")
    return audio, None


# ---- Stage implementations ----

def run_sed(args, rows, audio_dict):
    """SED inference via SEDInferenceStage: run PANNs CNN14, write NPZ."""
    from nemo_curator.stages.audio.inference.sed import SEDInferenceStage
    from nemo_curator.tasks import AudioTask

    stage = SEDInferenceStage(
        checkpoint_path=args.sed_checkpoint,
        model_type=args.sed_model_type,
        output_dir=args.output_dir,
    )
    stage.setup()
    print(f"SED model loaded ({args.sed_model_type})")

    for i, row in enumerate(rows):
        afp = row.get("audio_filepath", "")
        waveform_pair = audio_dict.get(afp) or audio_dict.get(os.path.basename(afp))
        if waveform_pair is None:
            row["sed_error"] = "audio not found in tar"
            continue

        # Write temp WAV for SED (it expects a file path)
        waveform, sr = waveform_pair
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform, sr)
            task = AudioTask(task_id=str(i), dataset_name="e2e", data={**row, "audio_filepath": tmp.name})
            try:
                result = stage.process(task)
                row.update({k: v for k, v in result.data.items() if k != "audio_filepath"})
            except Exception as e:
                print(f"  [{i}] SED error: {e}")
                row["sed_error"] = str(e)
            os.unlink(tmp.name)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}] SED processed")

    stage.teardown()
    return rows


def run_sed_post(args, rows, audio_dict):
    """SED postprocessing via SEDPostprocessingStage: NPZ -> speech events."""
    from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage
    from nemo_curator.tasks import AudioTask

    stage = SEDPostprocessingStage(speech_threshold=args.sed_threshold)
    stage.setup()

    for i, row in enumerate(rows):
        npz_path = row.get("npz_filepath")
        if not npz_path or not os.path.exists(npz_path):
            row["predicted_events"] = []
            continue
        task = AudioTask(task_id=str(i), dataset_name="e2e", data=dict(row))
        try:
            result = stage.process(task)
            row["predicted_events"] = result.data.get("predicted_events", [])
        except Exception as e:
            print(f"  [{i}] SED post error: {e}")
            row["predicted_events"] = []

    return rows


def run_segment(args, rows, audio_dict):
    """Segment extraction: fan-out speech events into individual entries."""
    output = []
    for row in rows:
        events = row.get("predicted_events", [])
        if not events:
            output.append(row)
            continue
        for j, ev in enumerate(events):
            seg = dict(row)
            seg["segment_start"] = ev.get("start_time", 0)
            seg["segment_end"] = ev.get("end_time", 0)
            seg["segment_idx"] = j
            seg.pop("predicted_events", None)
            output.append(seg)
    return output


def run_diarize(args, rows, audio_dict):
    """Diarization via InferenceSortformerStage."""
    from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage
    from nemo_curator.tasks import AudioTask

    stage = InferenceSortformerStage()
    stage.setup_on_node()
    stage.setup()
    print("Sortformer loaded")

    for i, row in enumerate(rows):
        afp = row.get("audio_filepath", "")
        waveform_pair = audio_dict.get(afp) or audio_dict.get(os.path.basename(afp))
        if waveform_pair is None:
            row["diar_segments"] = []
            continue

        waveform, sr = waveform_pair
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform, sr)
            task = AudioTask(task_id=str(i), dataset_name="e2e", data={**row, "audio_filepath": tmp.name})
            try:
                result = stage.process(task)
                row["diar_segments"] = result.data.get("diar_segments", [])
            except Exception as e:
                print(f"  [{i}] Diarize error: {e}")
                row["diar_segments"] = []
            os.unlink(tmp.name)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}] diarized")

    stage.teardown()
    return rows


def run_transcribe(args, rows, audio_dict):
    """3-pass transcription cascade via in-process vLLM."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    from nemo_curator.stages.audio.inference.transcription_cascade_inprocess import (
        TranscriptionCascadeInProcessStage,
    )

    stage = TranscriptionCascadeInProcessStage(
        model_id=args.omni_model,
        language=args.language,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        max_output_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    stage.setup()
    print(f"Transcription cascade loaded ({args.omni_model}, tp={args.tensor_parallel_size})")

    # Process in batches
    results = []
    from nemo_curator.tasks import AudioTask

    for i in range(0, len(rows), args.batch_size):
        batch_rows = rows[i:i + args.batch_size]
        tasks = []
        for j, row in enumerate(batch_rows):
            afp = row.get("audio_filepath", "")
            waveform_pair = audio_dict.get(afp) or audio_dict.get(os.path.basename(afp)) if audio_dict else None
            data = dict(row)
            if waveform_pair is not None:
                waveform, sr = waveform_pair
                data["waveform"] = waveform
                data["sample_rate"] = sr
            tasks.append(AudioTask(task_id=str(i + j), dataset_name="e2e", data=data))

        try:
            processed = stage.process_batch(tasks)
            for t in processed:
                result = {k: v for k, v in t.data.items()
                          if not isinstance(v, (np.ndarray, bytes, memoryview))}
                results.append(result)
        except Exception as e:
            print(f"  Batch {i // args.batch_size}: error: {e}")
            for row in batch_rows:
                row["transcribe_error"] = str(e)
                results.append(row)

        print(f"  Batch {(i // args.batch_size) + 1}: {len(results)} done")

    stage.teardown()
    return results


def run_embed(args, rows, audio_dict):
    """Speaker embedding extraction via TitaNet."""
    import torch
    import nemo.collections.asr as nemo_asr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        args.speaker_model, map_location=device
    )
    model.eval()
    print(f"TitaNet loaded on {device}")

    cut_ids = []
    all_embeddings = []
    batch_audios = []
    batch_ids = []

    for i, row in enumerate(rows):
        afp = row.get("audio_filepath", "")
        waveform_pair = audio_dict.get(afp) or audio_dict.get(os.path.basename(afp))
        if waveform_pair is None:
            continue
        waveform, sr = waveform_pair
        if sr != args.target_sr:
            try:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=args.target_sr)
            except ImportError:
                pass

        batch_audios.append(waveform)
        batch_ids.append(afp)

        if len(batch_audios) >= args.batch_size:
            embs = _embed_batch(model, batch_audios, device)
            all_embeddings.append(embs)
            cut_ids.extend(batch_ids)
            batch_audios.clear()
            batch_ids.clear()

    if batch_audios:
        embs = _embed_batch(model, batch_audios, device)
        all_embeddings.append(embs)
        cut_ids.extend(batch_ids)

    if all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0)
        os.makedirs(args.output_dir, exist_ok=True)
        np.savez(
            os.path.join(args.output_dir, f"embeddings_{args.shard_id}.npz"),
            cut_ids=np.array(cut_ids, dtype=object),
            embeddings=embeddings,
        )
        print(f"Saved {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")

    return rows


def _embed_batch(model, audios, device):
    import torch
    with torch.no_grad():
        return _embed_batch_inner(model, audios, device)


def _embed_batch_inner(model, audios, device):
    import torch
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


def run_group_video(args, rows, audio_dict):
    """Add video_id to each row."""
    from nemo_curator.stages.audio.preprocessing.group_by_video import extract_video_id
    for row in rows:
        row["video_id"] = extract_video_id(row)
    return rows


def run_cluster(args, rows, audio_dict):
    """Per-video speaker clustering. Reads embeddings from --embedding_dir."""
    from nemo_curator.stages.audio.preprocessing.group_by_video import extract_video_id
    from tutorials.audio.hifi_pipeline.slurm_speaker_id.cluster_by_video_v2 import (
        cluster_embeddings, speaker_confidence, l2_normalize,
    )
    from collections import defaultdict

    # Load embeddings
    emb_path = os.path.join(args.embedding_dir, f"embeddings_{args.shard_id}.npz")
    id_to_emb = {}
    if os.path.isfile(emb_path):
        data = np.load(emb_path, allow_pickle=True)
        for cid, emb in zip(data["cut_ids"], data["embeddings"].astype(np.float32)):
            id_to_emb[str(cid)] = emb

    # Group by video
    video_groups = defaultdict(list)
    for i, row in enumerate(rows):
        vid = extract_video_id(row)
        row["video_id"] = vid
        afp = row.get("audio_filepath", "")
        emb = id_to_emb.get(afp)
        video_groups[vid].append((i, row, emb))

    # Cluster per video
    for vid, entries in video_groups.items():
        valid = [(idx, r, e) for idx, r, e in entries if e is not None]
        if not valid:
            for idx, r, e in entries:
                r["speaker_label"] = -1
                r["confidence_score"] = 0.0
            continue

        emb_matrix = np.stack([e for _, _, e in valid])
        emb_matrix -= emb_matrix.mean(axis=0, keepdims=True)
        labels = cluster_embeddings(emb_matrix, args.cluster_threshold)
        scores = speaker_confidence(emb_matrix, labels)

        vid_prefix = abs(hash(vid)) % 1_000_000
        for j, (idx, r, e) in enumerate(valid):
            r["speaker_label"] = vid_prefix * 10000 + int(labels[j])
            r["confidence_score"] = round(float(scores[j]), 6)

        for idx, r, e in entries:
            if e is None:
                r["speaker_label"] = -1
                r["confidence_score"] = 0.0

    return rows


def _patch_utmosv2_for_gpu():
    import torch
    from utmosv2.dataset._base import BaseDataset
    from utmosv2.dataset._utils import extend_audio, select_random_start
    from utmosv2.dataset import SSLExtDataset
    import utmosv2.utils as utils
    import utmosv2._core.model._common as common

    class _GPURawAudioDataset(BaseDataset):
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

    _orig_get_dataset = utils.get_dataset

    def _patched_get_dataset(cfg, data, phase, transform=None):
        if cfg.dataset.name == "ssl_multispec_ext":
            return _GPURawAudioDataset(cfg, data, phase, transform)
        return _orig_get_dataset(cfg, data, phase, transform)

    utils.get_dataset = _patched_get_dataset
    common.get_dataset = _patched_get_dataset

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


def _build_gpu_spec_model(original_model, cfg):
    import torch
    import torch.nn as nn
    import torchaudio

    class _GPUSpecModel(nn.Module):
        def __init__(self, original_model, cfg):
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

        def _audio_to_spec(self, audio, spec_idx):
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

    return _GPUSpecModel(original_model, cfg)


def run_utmos(args, rows, audio_dict):
    """UTMOS scoring. Uses audio from tar (in-memory)."""
    _patch_utmosv2_for_gpu()
    import utmosv2
    model = utmosv2.create_model(pretrained=True)
    model._model = _build_gpu_spec_model(model._model, model._cfg)
    print("UTMOSv2 model loaded (GPU specs)")

    with tempfile.TemporaryDirectory() as wav_dir:
        valid_indices = []
        wav_count = 0

        for i, row in enumerate(rows):
            afp = row.get("audio_filepath", "")
            waveform_pair = audio_dict.get(afp) or audio_dict.get(os.path.basename(afp)) if audio_dict else None
            if waveform_pair is None:
                row["utmosv2_score"] = float("nan")
                continue

            waveform, sr = waveform_pair
            if sr != args.target_sr:
                try:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=args.target_sr)
                except ImportError:
                    pass

            wav_path = os.path.join(wav_dir, f"{wav_count:08d}.wav")
            sf.write(wav_path, waveform, args.target_sr, subtype="PCM_16")
            valid_indices.append(i)
            wav_count += 1

        print(f"Converted {wav_count} audio files to WAV")

        if valid_indices:
            t0 = time.time()
            results = model.predict(
                input_dir=wav_dir, batch_size=args.utmos_batch_size,
                num_repetitions=1, predict_dataset="sarulab",
                num_workers=0, verbose=False,
            )
            scores = [r["predicted_mos"] for r in results]
            for idx, score in zip(valid_indices, scores):
                rows[idx]["utmosv2_score"] = round(float(score), 4)
            print(f"Scored {len(scores)} files in {time.time()-t0:.1f}s")

    return rows


# ---- Dispatch ----

STAGES = {
    "sed": run_sed,
    "sed_post": run_sed_post,
    "segment": run_segment,
    "diarize": run_diarize,
    "transcribe": run_transcribe,
    "embed": run_embed,
    "group_video": run_group_video,
    "cluster": run_cluster,
    "utmos": run_utmos,
}

NEEDS_AUDIO = {"sed", "diarize", "transcribe", "embed", "utmos"}


def main():
    args = parse_args()
    t0 = time.time()

    # Skip if already done
    out_path = os.path.join(args.output_dir, f"shard_{args.shard_id}.jsonl")
    if args.stage != "embed" and os.path.exists(out_path):
        print(f"Already done: {out_path}")
        return
    if args.stage == "embed":
        emb_path = os.path.join(args.output_dir, f"embeddings_{args.shard_id}.npz")
        if os.path.exists(emb_path):
            print(f"Already done: {emb_path}")
            return

    print(f"Stage: {args.stage}, Shard: {args.shard_id}")
    rows = read_manifest(args.manifest_path)
    print(f"Loaded {len(rows)} rows from {args.manifest_path}")

    # Stream audio if needed
    audio_dict = None
    if args.stage in NEEDS_AUDIO and args.tar_url:
        audio_dict, _ = stream_tar_if_needed(args)

    # Run stage
    stage_fn = STAGES[args.stage]
    rows = stage_fn(args, rows, audio_dict)

    # Write output (embed writes its own npz)
    if args.stage != "embed":
        write_manifest(rows, args.output_dir, args.shard_id)
        print(f"Wrote {len(rows)} rows -> {out_path}")

    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
