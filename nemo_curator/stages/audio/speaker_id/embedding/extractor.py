"""Batched speaker-embedding extraction with duration-aware dynamic batching.

Sorts utterances **longest-first** so that any quadratic-scaling memory issues
(e.g. attention-based frontends) surface immediately rather than OOM-ing hours
into a run.  Accumulates a batch until total audio duration reaches
``batch_dur`` seconds, pads to max length, runs a single batched forward pass,
then collects per-utterance embeddings.
"""

import logging
import sys
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from nemo_curator.stages.audio.speaker_id.embedding.feature import compute_features, load_audio

logger = logging.getLogger(__name__)


def extract_embeddings(
    entries: List[dict],
    model: torch.nn.Module,
    frontend_type: str,
    device: torch.device,
    batch_dur: float = 600.0,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings for a list of utterance entries.

    Each entry must have ``_abs_wav_path``, ``_utt_id``, and optionally
    ``duration``.

    Args:
        entries: Utterance metadata dicts.
        model: WeSpeaker PyTorch model (eval mode, on *device*).
        frontend_type: ``"fbank"`` or a learned frontend name.
        device: Torch device the model lives on.
        batch_dur: Max cumulative audio seconds per dynamic batch.
        sample_rate: Target sample rate.
        num_mel_bins: Number of mel bins (fbank only).

    Returns:
        (embeddings_array, utt_ids) where embeddings_array has shape
        ``(N, emb_dim)`` and utt_ids is a list of N utterance IDs.
    """
    shard = sorted(entries, key=lambda e: e.get("duration", 0), reverse=True)

    utt_ids: List[str] = []
    embeddings: List[np.ndarray] = []
    failed = 0

    batch_feats: List[torch.Tensor] = []
    batch_utt_ids: List[str] = []
    batch_dur_acc = 0.0

    pbar = tqdm(shard, desc="Extracting", unit="utt", file=sys.stderr)
    for entry in pbar:
        wav_path = entry["_abs_wav_path"]
        utt_id = entry["_utt_id"]
        dur = entry.get("duration", 0)

        try:
            pcm = load_audio(wav_path, target_sr=sample_rate)
            feat = compute_features(
                pcm, model, frontend_type, device,
                sample_rate=sample_rate,
                num_mel_bins=num_mel_bins,
            )
            batch_feats.append(feat)
            batch_utt_ids.append(utt_id)
            batch_dur_acc += dur
        except Exception as e:
            failed += 1
            logger.warning("Failed loading %s: %s", utt_id, e)

        if batch_dur_acc >= batch_dur and batch_feats:
            _flush_batch(model, batch_feats, batch_utt_ids, embeddings, utt_ids, device)
            pbar.set_postfix(done=len(utt_ids), fail=failed)
            batch_feats.clear()
            batch_utt_ids.clear()
            batch_dur_acc = 0.0

    if batch_feats:
        _flush_batch(model, batch_feats, batch_utt_ids, embeddings, utt_ids, device)

    embeddings_np = np.vstack(embeddings) if embeddings else np.empty((0, 0))

    logger.info("Extracted %d embeddings (%d failed)", len(utt_ids), failed)
    return embeddings_np, utt_ids


def _flush_batch(
    model: torch.nn.Module,
    batch_feats: List[torch.Tensor],
    batch_utt_ids: List[str],
    embeddings_acc: List[np.ndarray],
    utt_ids_acc: List[str],
    device: torch.device,
) -> None:
    """Pad variable-length features, run batched forward, collect results."""
    lengths = [f.shape[0] for f in batch_feats]
    max_len = max(lengths)
    feat_dim = batch_feats[0].shape[1]

    padded = torch.zeros(len(batch_feats), max_len, feat_dim)
    for i, f in enumerate(batch_feats):
        padded[i, : f.shape[0], :] = f
    padded = padded.to(device)

    with torch.no_grad():
        outputs = model(padded)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs

    for i in range(len(batch_feats)):
        embeddings_acc.append(embeds[i].cpu().numpy())
        utt_ids_acc.append(batch_utt_ids[i])
