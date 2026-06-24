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

"""Long-form audio cutting pipeline for ALM pretraining data.

Builds a Curator ``Pipeline`` that takes a JSONL manifest of long-form
diarized + transcribed audio files plus a directory of source audio
files and produces:

* a tar archive of snippet audio files (mono, resampled), one tar
  member per snippet, stored at the tar root with no subdirectories
  (WebDataset/Energon-friendly)
* a JSONL manifest with one row per snippet (``snippet_id`` + segments
  with timestamps shifted to be snippet-relative)
* a metrics summary JSON with input/output counts, dropped-segment
  breakdowns, and a 30-second-bin histogram of snippet durations

Includes an n-gram-frequency repetition filter (between the planner and
the extractor) that drops snippets whose joined text shows Whisper-style
looping hallucinations, so filtered snippets never incur audio decode /
resample / file-write cost.

The output manifest is intended as the foundation for audio LLM
pretraining: each row's snippet-relative ``segments`` list is enough to
build interleaved audio/text continuation data, ASR training pairs, TTS
training pairs, or speaker-diarization training data without re-cutting
the source audio.
"""

from __future__ import annotations

from nemo_curator.pipeline import Pipeline

from .extraction import SnippetExtractionStage
from .finalize import (
    finalize_audio_pretrain_outputs,
    prepare_audio_pretrain_outputs,
)
from .io import (
    PretrainMetricsAggregatorStage,
    ReadLongFormManifestStage,
    SnippetManifestWriterStage,
)
from .planning import (
    OverlapFilterStage,
    SnippetCutPlannerStage,
    SnippetRepetitionFilterStage,
)
from .utils import AUDIO_PATH_RESOLUTION_BASENAME

__all__ = [
    "build_audio_pretrain_pipeline",
    "finalize_audio_pretrain_outputs",
    "prepare_audio_pretrain_outputs",
]


def build_audio_pretrain_pipeline(  # noqa: PLR0913
    *,
    input_manifest: str,
    audio_dir: str,
    output_dir: str,
    output_manifest_path: str,
    output_audio_tar_path: str,
    metrics_path: str,
    max_duration_sec: float,
    tokenizer_path: str,
    min_duration_sec: float = 0.5,
    min_overlap_sec: float = 0.5,
    # Default 30s: two segments separated by more than 30s of silence
    # are assumed to belong to semantically distinct conversations
    # (topic change, ad break, recording boundary). Grouping them into
    # one snippet would teach the model to associate unrelated content,
    # so the planner closes the snippet at long gaps even when the
    # max-duration budget would still permit appending.
    max_segment_gap_in_snippet: float = 30.0,
    ngram_n: int = 10,
    ngram_max_count: int = 3,
    tokenizer_cache_dir: str | None = None,
    hf_token: str | None = None,
    target_sample_rate: int = 16000,
    output_format: str = "flac",
    audio_filepath_key: str = "audio_filepath",
    audio_path_resolution: str = AUDIO_PATH_RESOLUTION_BASENAME,
    dataset_name: str = "long_form_audio",
    dry_run: bool = False,
) -> Pipeline:
    """Build the long-form-audio cutting pipeline for ALM pretraining.

    .. note::
       The writer and metrics aggregator stages each emit one shard file
       per replica, so the pipeline is safe under multi-replica backends
       (Xenna, Ray Data).  Callers MUST run
       :func:`prepare_audio_pretrain_outputs` once before
       :func:`Pipeline.run` and :func:`finalize_audio_pretrain_outputs`
       once after, on the driver, to clean up stale shards and merge
       this run's shards into the final user-visible files.  ``run.py``
       wires both calls automatically.

    Args:
        input_manifest: Path to the input JSONL manifest, one row per
            long-form audio.
        audio_dir: Directory containing the source audio files.  Each
            row's ``audio_filepath`` is re-anchored to this directory
            according to ``audio_path_resolution`` (default basename).
        output_dir: Directory where pipeline outputs are written.  The
            audio tar, manifest JSONL, and metrics JSON typically live
            here, though each has its own explicit path argument.
        output_manifest_path: Path of the output JSONL manifest (one row
            per snippet).  Each row's ``audio_filepath`` is the
            tar-internal basename of that snippet's audio member, not a
            filesystem path.
        output_audio_tar_path: Path of the output tar archive that
            contains every extracted snippet's audio file (one tar
            member per snippet, named ``<snippet_id>.<output_format>``).
            Members are stored at the tar root with no subdirectories,
            sorted lexicographically -- compatible with WebDataset and
            Energon tar-dataset readers.
        metrics_path: Path of the metrics summary JSON.
        max_duration_sec: Maximum snippet duration; greedy packing never
            exceeds this.  Single segments longer than this are dropped
            and counted under ``too_long``.
        min_duration_sec: Minimum snippet duration; shorter snippets are
            dropped and counted under ``too_short``.  Snippets whose
            concatenated text is empty are also dropped.
        min_overlap_sec: Two segments are considered overlapping (and
            both dropped) iff their intersection is at least this many
            seconds OR one fully contains the other.
        max_segment_gap_in_snippet: Maximum allowed silence (seconds)
            between two adjacent surviving segments inside a snippet.
            If the gap from the current snippet's last segment's ``end``
            to the next segment's ``start`` exceeds this, the planner
            closes the snippet and starts a new one even when
            ``max_duration_sec`` would still permit appending.  Default
            is 30s: segments separated by more than that are assumed to
            belong to semantically distinct conversations (topic change,
            ad break, recording boundary), which we don't want to bridge
            in a pretraining snippet.
        tokenizer_path: Either a local directory loadable by
            ``AutoTokenizer.from_pretrained`` or a HuggingFace Hub repo
            id (e.g. ``"openai/whisper-large-v3"``); when it's a repo
            id, the tokenizer is fetched once per node in
            ``setup_on_node`` so workers in ``setup`` only ever read
            from the local cache.  Used by the snippet repetition
            filter to detect Whisper-style looping hallucinations via
            n-gram frequency.
        ngram_n: N-gram size for the repetition filter; default 10.
        ngram_max_count: A snippet is dropped if any token-id n-gram in
            its joined text appears strictly more than this many times;
            default 3 (drop on ≥4 occurrences).  Filtered snippets are
            logged with the offending text highlighted in red.
        tokenizer_cache_dir: Optional ``cache_dir`` passed to
            ``snapshot_download`` and ``AutoTokenizer.from_pretrained``;
            ``None`` uses the HF default (``~/.cache/huggingface/hub``).
        hf_token: Optional HuggingFace token for gated tokenizer
            repositories; ``None`` uses the ambient HF auth state.
        target_sample_rate: Output snippet sample rate; the source audio
            is resampled with torchaudio if it differs.
        output_format: ``"wav"``, ``"flac"``, or ``"ogg"``.
        audio_filepath_key: JSONL field that holds the path to the audio
            file (default ``"audio_filepath"``).
        audio_path_resolution: How ``ReadLongFormManifestStage`` maps a
            row's ``audio_filepath`` to an on-disk path: ``"basename"``
            (default, with duplicate-basename detection), ``"relative"``
            (preserves subdirectories under ``audio_dir``), or
            ``"as_is"`` (trust the manifest's value).
        dataset_name: Tag stamped on emitted ``AudioTask`` objects.
        dry_run: If True, the extractor stage skips audio reads and
            writes -- no snippet audio files are produced -- but the
            output manifest and metrics summary are still generated.
            Useful for previewing how a real dataset would be cut.
            Snippet ``duration`` in the manifest will be the planned
            ``end - start`` (vs. the resampled-frame-count duration
            of a real run -- the difference is at most one frame at
            ``target_sample_rate``, ~62µs at 16 kHz).
    """
    return Pipeline(
        name="audio_pretrain_long_form_cut",
        description=("Long-form diarized audio -> bounded mono resampled snippets for audio LLM pretraining"),
        stages=[
            ReadLongFormManifestStage(
                input_manifest=input_manifest,
                audio_dir=audio_dir,
                audio_filepath_key=audio_filepath_key,
                audio_path_resolution=audio_path_resolution,
                dataset_name=dataset_name,
            ),
            OverlapFilterStage(min_overlap_sec=min_overlap_sec),
            SnippetCutPlannerStage(
                max_duration_sec=max_duration_sec,
                min_duration_sec=min_duration_sec,
                max_segment_gap_in_snippet=max_segment_gap_in_snippet,
            ),
            SnippetRepetitionFilterStage(
                tokenizer_path=tokenizer_path,
                ngram_n=ngram_n,
                ngram_max_count=ngram_max_count,
                cache_dir=tokenizer_cache_dir,
                hf_token=hf_token,
            ),
            SnippetExtractionStage(
                output_dir=output_dir,
                output_audio_tar_path=output_audio_tar_path,
                target_sample_rate=target_sample_rate,
                output_format=output_format,
                audio_filepath_key=audio_filepath_key,
                dry_run=dry_run,
            ),
            SnippetManifestWriterStage(output_path=output_manifest_path),
            PretrainMetricsAggregatorStage(output_path=metrics_path),
        ],
    )
