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

"""Derive stable shard keys from manifest paths for checkpointing and output layout.

Shard keys drive pipeline output directories, ``.jsonl`` manifests,
``.jsonl.done`` resume markers, and RTTM subdirs. They must be unique per
physical shard across languages and datasets sharing the same ``OUTPUT_DIR``.

YAML convention (``input_cfg`` entries)
---------------------------------------

**Preferred:** set ``shard_key_prefix`` when the S3 path does not embed a unique
corpus name, or when the same dataset folder name appears under multiple locales.
Use the catalog layout::

    <catalog-name>/<locale>/<dataset-id>/...

``corpus`` is then a logical catalog label (filtering / ``dataset_name``); it
need not appear in the manifest path. The anchor for tail extraction is the
*last* segment of ``shard_key_prefix`` (usually the dataset id or UUID).

Tarred example (known catalog, multi-bucket)::

    corpus: yt_harvested
    language: es
    shard_key_prefix: yt_harvested/es/youtube/v1.1_wer10_whisper/yt_mixed_2024_12_18_083625
    manifest_filepath: s3://asr/.../yt_mixed_.../bucket_OP_1..8_CL_/sharded_manifests/manifest__OP_0..4095_CL_.json
    tarred_audio_filepaths: s3://asr/.../yt_mixed_.../bucket_OP_1..8_CL_/audio__OP_0..4095_CL_.tar
    type: nemo_tarred

Tarred example (unknown corpus, UUID dataset id)::

    corpus: riva_de_batch
    language: de
    shard_key_prefix: riva_de_batch/de-DE/78e842f6-eac6-11ee-a616-03e701f9bfe1
    manifest_filepath: s3://asr/datasets/final/de-DE/78e842f6-.../sharded_manifests/manifest__OP_0..255_CL_.json
    tarred_audio_filepaths: s3://asr/datasets/final/de-DE/78e842f6-.../audio__OP_0..255_CL_.tar
    type: nemo_tarred

**Legacy:** omit ``shard_key_prefix`` only when ``corpus`` appears *exactly once*
as a path component (case-insensitive). Multi-language runs that share a dataset
folder name (e.g. ``yt_mixed_...`` under both ``de/`` and ``es/``) will collide
without a prefix — use separate ``OUTPUT_DIR`` per language or set ``shard_key_prefix``.
"""

from __future__ import annotations


def _strip_manifest_ext(rel: str) -> str:
    if rel.endswith(".jsonl.gz"):
        return rel[: -len(".jsonl.gz")]
    if rel.endswith(".jsonl"):
        return rel[: -len(".jsonl")]
    if rel.endswith(".json"):
        return rel[: -len(".json")]
    return rel


def derive_manifest_shard_key(
    manifest_path: str,
    corpus: str,
    *,
    shard_key_prefix: str | None = None,
) -> str:
    """Derive a shard key from a manifest path.

    Behavior:
    - If ``shard_key_prefix`` is set (via the YAML), the shard key is
      ``{prefix}/{tail}`` where ``tail`` is the portion of the manifest path
      *after* the prefix's last segment (the "anchor"). This preserves the
      distinguishing components (e.g. ``bucket_5/sharded_manifests/manifest_000042``)
      so shards from different buckets/languages never collide. If the anchor is
      not found in the path, it falls back to ``{prefix}/{manifest_basename}``.
      This is the escape hatch for datasets whose paths don't embed the corpus
      name (or embed it more than once).
    - Otherwise, the corpus name must appear exactly once in the path; the shard
      key is everything from that component onward, with the extension stripped.
    """
    parts = manifest_path.replace("\\", "/").split("/")
    parts[-1] = _strip_manifest_ext(parts[-1])
    parts_lower = [p.lower() for p in parts]

    if shard_key_prefix:
        prefix = shard_key_prefix.strip("/")
        if not prefix:
            return parts[-1]
        anchor = prefix.split("/")[-1].lower()
        # Use the last occurrence of the anchor so the tail is closest to the manifest.
        anchor_idx = next((i for i in range(len(parts_lower) - 1, -1, -1) if parts_lower[i] == anchor), None)
        if anchor_idx is not None:
            tail = parts[anchor_idx + 1 :]
            return f"{prefix}/{'/'.join(tail)}" if tail else prefix
        # Anchor not in path — keep only the manifest basename after the prefix.
        return f"{prefix}/{parts[-1]}"

    corpus_lower = corpus.lower()
    matches = [i for i, p in enumerate(parts_lower) if p == corpus_lower]
    if len(matches) == 0:
        msg = (
            f"Corpus name '{corpus}' not found in manifest path: {manifest_path}. "
            f"The YAML 'corpus' field must match a directory component in the manifest path "
            f"(case-insensitive), or set 'shard_key_prefix' in the YAML."
        )
        raise ValueError(msg)
    if len(matches) > 1:
        msg = (
            f"Corpus name '{corpus}' appears {len(matches)} times in manifest path: {manifest_path}. "
            f"It must appear exactly once for unambiguous path extraction, or set "
            f"'shard_key_prefix' in the YAML."
        )
        raise ValueError(msg)
    return "/".join(parts[matches[0] :])
