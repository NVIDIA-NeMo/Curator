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

* a directory of snippet audio files (mono, resampled), one per snippet
* a JSONL manifest with one row per snippet (``snippet_id`` + segments
  with timestamps shifted to be snippet-relative)
* a metrics summary JSON with input/output counts, dropped-segment
  breakdowns, and a 30-second-bin histogram of snippet durations

The output manifest is intended as the foundation for audio LLM
pretraining: each row's snippet-relative ``segments`` list is enough to
build interleaved audio/text continuation data, ASR training pairs, TTS
training pairs, or speaker-diarization training data without re-cutting
the source audio.
"""

from __future__ import annotations

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.pretrain import (
    OverlapFilterStage,
    PretrainMetricsAggregatorStage,
    ReadLongFormManifestStage,
    SnippetCutPlannerStage,
    SnippetExtractionStage,
    SnippetManifestWriterStage,
    finalize_audio_pretrain_outputs,
    prepare_audio_pretrain_outputs,
)

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
    metrics_path: str,
    max_duration_sec: float,
    min_duration_sec: float = 0.5,
    min_overlap_sec: float = 0.5,
    # Default 30s: two segments separated by more than 30s of silence
    # are assumed to belong to semantically distinct conversations
    # (topic change, ad break, recording boundary). Grouping them into
    # one snippet would teach the model to associate unrelated content,
    # so the planner closes the snippet at long gaps even when the
    # max-duration budget would still permit appending.
    max_segment_gap_in_snippet: float = 30.0,
    target_sample_rate: int = 16000,
    output_format: str = "flac",
    audio_filepath_key: str = "audio_filepath",
    dataset_name: str = "long_form_audio",
    dry_run: bool = False,
) -> Pipeline:
    """Build the long-form-audio cutting pipeline for ALM pretraining.

    .. note::
       The writer and metrics aggregator stages each emit one shard file
       per replica, so the pipeline is safe under multi-replica backends
       (Xenna, Ray Data, Ray Actor Pool).  Callers MUST run
       :func:`prepare_audio_pretrain_outputs` once before
       :func:`Pipeline.run` and :func:`finalize_audio_pretrain_outputs`
       once after, on the driver, to clean up stale shards and merge
       this run's shards into the final user-visible files.  ``run.py``
       wires both calls automatically.

    Args:
        input_manifest: Path to the input JSONL manifest, one row per
            long-form audio.
        audio_dir: Directory containing the source audio files.  Each
            row's ``audio_filepath`` is re-anchored to this directory by
            basename.
        output_dir: Directory where snippet audio files are written.
        output_manifest_path: Path of the output JSONL manifest (one row
            per snippet).
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
        target_sample_rate: Output snippet sample rate; the source audio
            is resampled with torchaudio if it differs.
        output_format: ``"wav"``, ``"flac"``, or ``"ogg"``.
        audio_filepath_key: JSONL field that holds the path to the audio
            file (default ``"audio_filepath"``).
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
                dataset_name=dataset_name,
            ),
            OverlapFilterStage(min_overlap_sec=min_overlap_sec),
            SnippetCutPlannerStage(
                max_duration_sec=max_duration_sec,
                min_duration_sec=min_duration_sec,
                max_segment_gap_in_snippet=max_segment_gap_in_snippet,
            ),
            SnippetExtractionStage(
                output_dir=output_dir,
                target_sample_rate=target_sample_rate,
                output_format=output_format,
                audio_filepath_key=audio_filepath_key,
                dry_run=dry_run,
            ),
            SnippetManifestWriterStage(output_path=output_manifest_path),
            PretrainMetricsAggregatorStage(output_path=metrics_path),
        ],
    )
