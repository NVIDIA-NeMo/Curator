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

"""Stage-level tests for ``nemo_curator.stages.audio.alm.pretrain.planning``.

Covers overlap filtering, snippet planning, and the n-gram repetition
filter.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from nemo_curator.stages.audio.alm.pretrain import (
    OverlapFilterStage,
    SnippetCutPlannerStage,
    SnippetRepetitionFilterStage,
)
from nemo_curator.stages.audio.alm.pretrain.utils import (
    _PLAN_DATA_KEY,
    _PRETRAIN_META_KEY,
)
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


def _make_audio_task(data: dict | None = None, *, task_id: str = "t1") -> AudioTask:
    return AudioTask(task_id=task_id, dataset_name="ds", data=data or {})


def _ts(start: float, end: float, text: str = "x", text_itn: str | None = None) -> dict:
    return {
        "speaker": "A",
        "start": start,
        "end": end,
        "text": text,
        "text_ITN": text_itn if text_itn is not None else text,
        "words": [],
    }


# ----------------------------------------------------------------------
# OverlapFilterStage
# ----------------------------------------------------------------------


class TestOverlapFilterStage:
    def test_drops_empty_then_overlap(self) -> None:
        segs = [
            _ts(0, 3, "a"),  # ok
            {"start": 5, "end": 6, "text": "", "text_ITN": "", "words": []},  # empty
            _ts(10, 15, "b"),  # overlap pair
            _ts(13, 18, "c"),  # overlap pair
            _ts(20, 23, "d"),  # ok
        ]
        task = _make_audio_task({"id": "X", "segments": segs})
        stage = OverlapFilterStage()
        out = stage.process(task)
        # Kept: only "a" and "d"
        assert [s["text"] for s in out.data["segments"]] == ["a", "d"]
        meta = out._metadata[_PRETRAIN_META_KEY]
        assert meta["original_seg_count"] == 5
        assert meta["dropped_empty"] == 1
        assert meta["dropped_overlap"] == 2
        assert meta["kept_after_filter_count"] == 2

    def test_no_segments_metadata_initialized(self) -> None:
        task = _make_audio_task({"id": "X", "segments": []})
        stage = OverlapFilterStage()
        out = stage.process(task)
        meta = out._metadata[_PRETRAIN_META_KEY]
        assert meta["original_seg_count"] == 0
        assert meta["dropped_empty"] == 0
        assert meta["dropped_overlap"] == 0


# ----------------------------------------------------------------------
# SnippetCutPlannerStage
# ----------------------------------------------------------------------


class TestSnippetCutPlannerStage:
    def test_writes_plan_and_drop_counts(self) -> None:
        segs = [_ts(0, 5, "a"), _ts(5, 10, "b")]
        task = _make_audio_task({"id": "X", "segments": segs})
        stage = SnippetCutPlannerStage(max_duration_sec=20.0, min_duration_sec=0.5, max_segment_gap_in_snippet=30.0)
        out = stage.process(task)
        plan = out.data[_PLAN_DATA_KEY]
        assert len(plan) == 1
        assert (plan[0]["start"], plan[0]["end"]) == (0, 10)
        meta = out._metadata[_PRETRAIN_META_KEY]
        assert meta["planned_snippets"] == 1
        assert meta["dropped_too_long"] == 0
        assert meta["dropped_too_short"] == 0
        assert meta["dropped_no_text"] == 0

    def test_invalid_args_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_duration"):
            SnippetCutPlannerStage(max_duration_sec=-1.0).__post_init__()
        with pytest.raises(ValueError, match="min_duration"):
            SnippetCutPlannerStage(min_duration_sec=-1.0).__post_init__()
        with pytest.raises(ValueError, match="min_duration_sec must be <="):
            SnippetCutPlannerStage(max_duration_sec=5.0, min_duration_sec=10.0).__post_init__()
        with pytest.raises(ValueError, match="max_segment_gap_in_snippet"):
            SnippetCutPlannerStage(max_segment_gap_in_snippet=-0.1).__post_init__()


# ----------------------------------------------------------------------
# SnippetRepetitionFilterStage
# ----------------------------------------------------------------------


def _build_tiny_word_tokenizer(tmp_dir: Path, words: list[str]) -> Path:
    """Save a WordLevel HF fast tokenizer covering ``words`` to ``tmp_dir``."""
    vocab = {"[UNK]": 0, **{w: i for i, w in enumerate(words, start=1)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))  # noqa: S106  -- tokenizer special token, not a credential
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_dir = tmp_dir / "tok"
    tok_dir.mkdir()
    tok.save(str(tok_dir / "tokenizer.json"))
    (tok_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 4096}),
        encoding="utf-8",
    )
    return tok_dir


class TestSnippetRepetitionFilterStage:
    @pytest.fixture
    def tokenizer_dir(self, tmp_path: Path) -> Path:
        pytest.importorskip("transformers")
        pytest.importorskip("tokenizers")
        return _build_tiny_word_tokenizer(
            tmp_path,
            ["thank", "you", "for", "watching", "please", "subscribe", "the", "quick", "brown", "fox", "hi"],
        )

    def _make_task_with_plan(self, plan: list[dict]) -> AudioTask:
        task = AudioTask(task_id="t1", dataset_name="ds", data={_PLAN_DATA_KEY: plan})
        task._metadata = {}
        return task

    def test_drops_repetitive_snippet(self, tokenizer_dir: Path) -> None:
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir))
        stage.setup()

        repeat = "thank you for watching " * 10
        plan = [{"start": 0.0, "end": 30.0, "segments": [_ts(0.0, 30.0, repeat)]}]
        out = stage.process(self._make_task_with_plan(plan))
        assert out.data[_PLAN_DATA_KEY] == []
        meta = out._metadata[_PRETRAIN_META_KEY]
        assert meta["dropped_repetition"] == 1
        assert meta["kept_after_repetition_filter"] == 0
        # The dropped text is captured (un-colorized) for the metrics summary.
        assert meta["filtered_repetition_texts"] == [repeat.strip()]

    def test_keeps_non_repetitive_snippet(self, tokenizer_dir: Path) -> None:
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir))
        stage.setup()

        plan = [
            {"start": 0.0, "end": 5.0, "segments": [_ts(0.0, 5.0, "the quick brown fox")]},
        ]
        out = stage.process(self._make_task_with_plan(plan))
        assert len(out.data[_PLAN_DATA_KEY]) == 1
        meta = out._metadata[_PRETRAIN_META_KEY]
        assert meta["dropped_repetition"] == 0
        assert meta["kept_after_repetition_filter"] == 1

    def test_keeps_short_snippet_without_enough_tokens_for_ngram(self, tokenizer_dir: Path) -> None:
        # ngram_n=4 but text tokenizes to 1 token -> no n-grams to evaluate, kept
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir), ngram_n=4)
        stage.setup()

        plan = [{"start": 0.0, "end": 1.0, "segments": [_ts(0.0, 1.0, "hi")]}]
        out = stage.process(self._make_task_with_plan(plan))
        assert len(out.data[_PLAN_DATA_KEY]) == 1
        assert out._metadata[_PRETRAIN_META_KEY]["dropped_repetition"] == 0

    def test_filters_only_repetitive_snippets_in_a_mixed_plan(self, tokenizer_dir: Path) -> None:
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir))
        stage.setup()

        plan = [
            {"start": 0.0, "end": 5.0, "segments": [_ts(0.0, 5.0, "the quick brown fox")]},
            {"start": 5.0, "end": 35.0, "segments": [_ts(5.0, 35.0, "thank you for watching " * 10)]},
            {"start": 35.0, "end": 36.0, "segments": [_ts(35.0, 36.0, "hi")]},
        ]
        out = stage.process(self._make_task_with_plan(plan))
        kept_texts = [s["segments"][0]["text"] for s in out.data[_PLAN_DATA_KEY]]
        assert kept_texts == ["the quick brown fox", "hi"]
        assert out._metadata[_PRETRAIN_META_KEY]["dropped_repetition"] == 1
        assert out._metadata[_PRETRAIN_META_KEY]["kept_after_repetition_filter"] == 2
        # The override of planned_snippets reflects the post-filter count.
        assert out._metadata[_PRETRAIN_META_KEY]["planned_snippets"] == 2

    def test_post_init_validates(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ngram_n"):
            SnippetRepetitionFilterStage(tokenizer_path=str(tmp_path), ngram_n=0).__post_init__()
        with pytest.raises(ValueError, match="ngram_max_count"):
            SnippetRepetitionFilterStage(tokenizer_path=str(tmp_path), ngram_max_count=0).__post_init__()

    def test_per_source_example_cap(self, tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The per-source filtered_repetition_texts list is capped."""
        import nemo_curator.stages.audio.alm.pretrain.planning as m

        monkeypatch.setattr(m, "_MAX_FILTERED_TEXT_EXAMPLES", 3)
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir))
        stage.setup()

        repeat = "thank you for watching " * 10
        plan = [{"start": float(i), "end": float(i) + 30.0, "segments": [_ts(0.0, 30.0, repeat)]} for i in range(7)]
        out = stage.process(self._make_task_with_plan(plan))
        meta = out._metadata[_PRETRAIN_META_KEY]
        # All 7 are counted as dropped, but only the first 3 texts are retained.
        assert meta["dropped_repetition"] == 7
        assert len(meta["filtered_repetition_texts"]) == 3

    def test_process_is_idempotent_under_re_execution(self, tokenizer_dir: Path) -> None:
        """Calling process() twice on the same source must not double-count.

        Ray Data may fan a stage out across multiple blocks for the same
        upstream task, so process() can run more than once per source. The
        per-source counters and example list must be assignment-based so
        the second run overwrites rather than appends.
        """
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir))
        stage.setup()

        repeat = "thank you for watching " * 10
        # Build TWO tasks with the same plan; process() runs once per task.
        task1 = self._make_task_with_plan(
            [{"start": 0.0, "end": 30.0, "segments": [_ts(0.0, 30.0, repeat)]}]
        )
        task2 = self._make_task_with_plan(
            [{"start": 0.0, "end": 30.0, "segments": [_ts(0.0, 30.0, repeat)]}]
        )
        # First-pass result.
        out1 = stage.process(task1)
        meta1 = out1._metadata[_PRETRAIN_META_KEY]
        first_count = meta1["dropped_repetition"]
        first_texts = list(meta1["filtered_repetition_texts"])
        # Second pass on the same metadata dict (simulates re-execution).
        # We feed it a task that ALREADY carries the prior-pass metadata.
        task2._metadata[_PRETRAIN_META_KEY] = dict(meta1)
        out2 = stage.process(task2)
        meta2 = out2._metadata[_PRETRAIN_META_KEY]
        # Counters identical (overwrite semantics, not append).
        assert meta2["dropped_repetition"] == first_count
        assert meta2["filtered_repetition_texts"] == first_texts
