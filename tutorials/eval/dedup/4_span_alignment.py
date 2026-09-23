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

"""
Step 4 of the fuzzy-dedup-eval example: add span-alignment evidence to the
labeled pairs written by 3_build_pair_dataset.py.

Example:
    python tutorials/eval/dedup/4_span_alignment.py \
        --input-path output/dedup_eval/keeper_removed_pairs_raw \
        --output-path output/dedup_eval/keeper_removed_pairs
"""

from __future__ import annotations

import argparse
import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.tasks import DocumentBatch

# Keep in sync with judge_config/fuzzy_pair_judge.yaml's dynamo_model.engine_kwargs.max_model_len --
# this is how much of each document the judge actually sees.
MAX_VISIBLE_CHARS = 6000
# Pad the true start/end of an A-only/B-only difference with this many neighboring characters of
# shared text so a short changed value (an id, price, or date) isn't shown to the judge in
# isolation -- see README.md's "Span alignment and truncation" section for the "Applicable to
# Model X100/X200" example this is meant to cover. Only applied to the outer edges of a
# difference, not between the chunks _MAX_SPAN_CHUNK_CHARS splits a long difference into.
_SPAN_CONTEXT_CHARS = 40
# Split any single span longer than this into multiple spans, so one long difference (or one long
# run of unchanged text) doesn't become a single undifferentiated block the judge has to read as
# a whole -- matches the token-boundary chunking used by NeMo Curator's own dedup-eval judging
# pipeline (`_chunk_token_range`/`_MAX_DIFF_SPAN_CHARS`).
_MAX_SPAN_CHUNK_CHARS = 220
# Cap spans per kind so one wildly different pair can't blow up the rendered prompt; matches
# fuzzy_pair_judge.yaml's max_tokens budget. A pair that exceeds this gets status
# INCOMPLETE_LIMIT instead of a silently truncated span list. Applied after chunking, since
# chunking is what determines the final span count.
_MAX_SPANS_PER_KIND = 160
_DIFF_TOKEN_PATTERN = re.compile(r"\w{1,120}|[^\w\s]", re.UNICODE)

Token = tuple[int, int, str]


def _normalized_diff_tokens(text: str) -> list[Token]:
    return [
        (match.start(), match.end(), unicodedata.normalize("NFKC", match.group()).casefold())
        for match in _DIFF_TOKEN_PATTERN.finditer(text)
    ]


def _tokenize_pair(text_a: str | None, text_b: str | None, *, max_chars: int) -> dict[str, Any]:
    text_a, text_b = text_a or "", text_b or ""
    visible_a, visible_b = text_a[:max_chars], text_b[:max_chars]
    return {
        "tokens_a": _normalized_diff_tokens(visible_a),
        "tokens_b": _normalized_diff_tokens(visible_b),
        "truncated_a": len(text_a) > max_chars,
        "truncated_b": len(text_b) > max_chars,
    }


def _align_tokens(tokens_a: list[Token], tokens_b: list[Token]) -> list[dict[str, Any]]:
    """Diff two token streams into SHARED/A_ONLY/B_ONLY segments, as token-index ranges.

    Segments are left unchunked/unpadded/unsliced -- `SpanChunkingStage` re-slices `tokens_a`/
    `tokens_b` at these indices instead of re-tokenizing.
    """
    normalized_a = [token[2] for token in tokens_a]
    normalized_b = [token[2] for token in tokens_b]
    matcher = difflib.SequenceMatcher(a=normalized_a, b=normalized_b, autojunk=False)
    segments = []
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag == "equal" and a_end > a_start:
            segments.append(
                {
                    "kind": "SHARED",
                    "a_tok_start": a_start,
                    "a_tok_end": a_end,
                    "b_tok_start": b_start,
                    "b_tok_end": b_end,
                }
            )
            continue
        if tag in ("delete", "replace") and a_end > a_start:
            segments.append({"kind": "A_ONLY", "a_tok_start": a_start, "a_tok_end": a_end})
        if tag in ("insert", "replace") and b_end > b_start:
            segments.append({"kind": "B_ONLY", "b_tok_start": b_start, "b_tok_end": b_end})
    return segments


def _chunk_token_range(
    tokens: list[Token], start: int, end: int, *, paired: list[Token] | None = None
) -> list[tuple[int, int]]:
    """Split token indices [start, end) into chunks no longer than `_MAX_SPAN_CHUNK_CHARS`.

    If `paired` is given (the token-for-token-aligned range from the other side of a SHARED
    segment), a chunk also ends once the paired side would exceed the limit, so the two sides
    stay in lockstep.
    """
    chunks = []
    cursor = start
    while cursor < end:
        chunk_start = cursor
        while cursor < end:
            span_length = tokens[cursor][1] - tokens[chunk_start][0]
            paired_length = paired[cursor - start][1] - paired[chunk_start - start][0] if paired is not None else 0
            if cursor > chunk_start and max(span_length, paired_length) > _MAX_SPAN_CHUNK_CHARS:
                break
            cursor += 1
        chunks.append((chunk_start, cursor))
    return chunks


def _chunk_segments(  # noqa: PLR0913
    segments: list[dict[str, Any]],
    *,
    tokens_a: list[Token],
    tokens_b: list[Token],
    text_a: str | None,
    text_b: str | None,
    truncated_a: bool,
    truncated_b: bool,
    max_chars: int,
) -> dict[str, Any]:
    """Chunk/pad diff segments into the final SHARED/A_ONLY/B_ONLY spans and assemble the packet."""
    text_a, text_b = text_a or "", text_b or ""
    visible_len_a, visible_len_b = min(max_chars, len(text_a)), min(max_chars, len(text_b))
    raw_spans: dict[str, list[dict[str, Any]]] = {"SHARED": [], "A_ONLY": [], "B_ONLY": []}

    for segment in segments:
        kind = segment["kind"]
        if kind == "SHARED":
            a_start, a_end = segment["a_tok_start"], segment["a_tok_end"]
            b_start = segment["b_tok_start"]
            paired_tokens = tokens_b[b_start : segment["b_tok_end"]]
            for chunk_start, chunk_end in _chunk_token_range(tokens_a, a_start, a_end, paired=paired_tokens):
                paired_start = b_start + (chunk_start - a_start)
                paired_end = paired_start + (chunk_end - chunk_start)
                a_start_char, a_end_char = tokens_a[chunk_start][0], tokens_a[chunk_end - 1][1]
                b_start_char, b_end_char = tokens_b[paired_start][0], tokens_b[paired_end - 1][1]
                raw_spans["SHARED"].append(
                    {
                        "a_start_char": a_start_char,
                        "a_end_char": a_end_char,
                        "a_text": text_a[a_start_char:a_end_char],
                        "b_start_char": b_start_char,
                        "b_end_char": b_end_char,
                        "b_text": text_b[b_start_char:b_end_char],
                    }
                )
            continue

        side, tokens, text, visible_len = (
            ("A", tokens_a, text_a, visible_len_a) if kind == "A_ONLY" else ("B", tokens_b, text_b, visible_len_b)
        )
        tok_start, tok_end = segment[f"{side.lower()}_tok_start"], segment[f"{side.lower()}_tok_end"]
        chunks = _chunk_token_range(tokens, tok_start, tok_end)
        for index, (chunk_start, chunk_end) in enumerate(chunks):
            start_char, end_char = tokens[chunk_start][0], tokens[chunk_end - 1][1]
            if index == 0:
                start_char = max(0, start_char - _SPAN_CONTEXT_CHARS)
            if index == len(chunks) - 1:
                end_char = min(visible_len, end_char + _SPAN_CONTEXT_CHARS)
            raw_spans[kind].append({"start_char": start_char, "end_char": end_char, "text": text[start_char:end_char]})

    counts = {kind: len(items) for kind, items in raw_spans.items()}
    complete = all(count <= _MAX_SPANS_PER_KIND for count in counts.values())
    prefixes = {"SHARED": "S", "A_ONLY": "A", "B_ONLY": "B"}
    spans = []
    for kind in ("SHARED", "A_ONLY", "B_ONLY"):
        for index, item in enumerate(raw_spans[kind][:_MAX_SPANS_PER_KIND], start=1):
            spans.append({"span_id": f"{prefixes[kind]}{index:03d}", "kind": kind, **item})

    return {
        "status": "COMPLETE" if complete else "INCOMPLETE_LIMIT",
        "truncated": truncated_a or truncated_b,
        "truncated_a": truncated_a,
        "truncated_b": truncated_b,
        "span_counts": counts,
        "spans": spans,
    }


def build_semantic_diff(
    text_a: str | None, text_b: str | None, *, max_chars: int = MAX_VISIBLE_CHARS
) -> dict[str, Any]:
    """Align document A and B into SHARED/A_ONLY/B_ONLY spans with stable IDs.

    Convenience wrapper composing `_tokenize_pair()` -> `_align_tokens()` -> `_chunk_segments()`
    for a single pair outside the Ray Data pipeline (e.g. for tests). The pipeline itself runs
    these as three separate stages -- see the module docstring.
    """
    tokenized = _tokenize_pair(text_a, text_b, max_chars=max_chars)
    segments = _align_tokens(tokenized["tokens_a"], tokenized["tokens_b"])
    return _chunk_segments(
        segments,
        tokens_a=tokenized["tokens_a"],
        tokens_b=tokenized["tokens_b"],
        text_a=text_a,
        text_b=text_b,
        truncated_a=tokenized["truncated_a"],
        truncated_b=tokenized["truncated_b"],
        max_chars=max_chars,
    )


@dataclass
class TokenizerStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Tokenizes the judge-visible (truncated) `text_a`/`text_b` once per pair.

    `SpanAlignmentStage` and `SpanChunkingStage` both reuse these token boundaries instead of
    re-tokenizing.
    """

    max_chars: int = MAX_VISIBLE_CHARS
    name: str = "span_tokenizer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text_a", "text_b"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["tokens_a", "tokens_b", "truncated_a", "truncated_b"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        tokenized = [
            _tokenize_pair(row["text_a"], row["text_b"], max_chars=self.max_chars) for row in df.to_dict("records")
        ]
        df = df.assign(
            tokens_a=[t["tokens_a"] for t in tokenized],
            tokens_b=[t["tokens_b"] for t in tokenized],
            truncated_a=[t["truncated_a"] for t in tokenized],
            truncated_b=[t["truncated_b"] for t in tokenized],
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name, data=df, _metadata=batch._metadata, _stage_perf=batch._stage_perf
        )


@dataclass
class SpanAlignmentStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Diffs `tokens_a`/`tokens_b` (from `TokenizerStage`) into SHARED/A_ONLY/B_ONLY segments.

    Segments are token-index ranges, not yet char-sliced, padded, or chunked -- `SpanChunkingStage`
    does that against the same token lists.
    """

    name: str = "span_alignment"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["tokens_a", "tokens_b"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["diff_segments"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        segments = [_align_tokens(row["tokens_a"], row["tokens_b"]) for row in df.to_dict("records")]
        df = df.assign(diff_segments=segments)
        return DocumentBatch(
            dataset_name=batch.dataset_name, data=df, _metadata=batch._metadata, _stage_perf=batch._stage_perf
        )


@dataclass
class SpanChunkingStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Chunks/pads `diff_segments` into the final `semantic_diff`/`truncated` columns.

    Re-slices `tokens_a`/`tokens_b` at the token indices `SpanAlignmentStage` already computed --
    no re-tokenization. Drops the intermediate `tokens_a`/`tokens_b`/`diff_segments`/
    `truncated_a`/`truncated_b` columns so they don't leak into the written pairs files.
    """

    max_chars: int = MAX_VISIBLE_CHARS
    name: str = "span_chunking"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["diff_segments", "tokens_a", "tokens_b", "text_a", "text_b", "truncated_a", "truncated_b"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["semantic_diff", "truncated"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        diffs = [
            _chunk_segments(
                row["diff_segments"],
                tokens_a=row["tokens_a"],
                tokens_b=row["tokens_b"],
                text_a=row["text_a"],
                text_b=row["text_b"],
                truncated_a=row["truncated_a"],
                truncated_b=row["truncated_b"],
                max_chars=self.max_chars,
            )
            for row in df.to_dict("records")
        ]
        df = df.assign(semantic_diff=diffs, truncated=[diff["truncated"] for diff in diffs])
        df = df.drop(columns=["tokens_a", "tokens_b", "diff_segments", "truncated_a", "truncated_b"])
        return DocumentBatch(
            dataset_name=batch.dataset_name, data=df, _metadata=batch._metadata, _stage_perf=batch._stage_perf
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-path",
        required=True,
        help="Directory of raw pairs part_*.jsonl files written by 3_build_pair_dataset.py.",
    )
    parser.add_argument(
        "--output-path", required=True, help="Directory to write the span-aligned pairs JSONL files to."
    )
    parser.add_argument(
        "--files-per-partition",
        type=int,
        default=1,
        help="Raw pairs files grouped into each reader task, shared by TokenizerStage/SpanAlignmentStage/"
        "SpanChunkingStage. Default 1 preserves 3_build_pair_dataset.py's --pairs-per-file sharding as the "
        "parallelism unit for both this step and 5_run_llm_judge.py.",
    )
    parser.add_argument(
        "--max-visible-chars",
        type=int,
        default=MAX_VISIBLE_CHARS,
        help="Characters of text_a/text_b to align/show the judge per side. MUST match "
        "judge_config/pair.jinja's truncation (currently also 6000) and stay within "
        "judge_config/fuzzy_pair_judge.yaml's max_model_len.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline = Pipeline(
        name="dedup_eval_span_alignment",
        description="Add semantic-diff span-alignment evidence to labeled pairs.",
        stages=[
            JsonlReader(file_paths=args.input_path, files_per_partition=args.files_per_partition),
            TokenizerStage(max_chars=args.max_visible_chars),
            SpanAlignmentStage(),
            SpanChunkingStage(max_chars=args.max_visible_chars),
            JsonlWriter(path=args.output_path, mode="overwrite"),
        ],
    )

    ray_client = RayClient()
    ray_client.start()
    try:
        pipeline.run()
    finally:
        ray_client.stop()

    logger.info(f"Wrote span-aligned pairs to {args.output_path}")


if __name__ == "__main__":
    main()
