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

"""Pure-helper unit tests for the pretrain pipeline.

No Curator stage machinery, no soundfile / torch.  These tests cover the
algorithm helpers that the stages in ``planning`` / ``utils`` /
``finalize`` call into.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from nemo_curator.stages.audio.alm.pretrain.finalize import _build_final_summary
from nemo_curator.stages.audio.alm.pretrain.planning import (
    _count_ngrams,
    _find_offending_ngrams,
    _format_red,
    _locate_ngram_char_ranges,
    _merge_char_ranges,
    filter_empty_segments,
    find_overlapping_indices,
    plan_snippets,
    relativize_segments,
)
from nemo_curator.stages.audio.alm.pretrain.utils import (
    _delete_shards,
    _glob_shards,
    _make_shard_path,
    _resolve_audio_path,
    _segment_text,
    histogram_30s,
    make_snippet_id,
)

# ----------------------------------------------------------------------
# _segment_text
# ----------------------------------------------------------------------


class TestSegmentText:
    def test_returns_text_field(self) -> None:
        assert _segment_text({"text": "hello"}) == "hello"

    def test_strips_whitespace(self) -> None:
        assert _segment_text({"text": "  hi  "}) == "hi"

    def test_returns_empty_when_text_missing(self) -> None:
        assert _segment_text({}) == ""

    def test_returns_empty_when_text_empty(self) -> None:
        assert _segment_text({"text": ""}) == ""

    def test_ignores_text_itn(self) -> None:
        # text_ITN is unreliable in real upstream data; the helper only reads `text`.
        assert _segment_text({"text": "", "text_ITN": "ignored"}) == ""
        assert _segment_text({"text": "kept", "text_ITN": "Other."}) == "kept"


# ----------------------------------------------------------------------
# filter_empty_segments
# ----------------------------------------------------------------------


class TestFilterEmptySegments:
    def test_keeps_segments_with_text(self) -> None:
        segs = [{"text": "a", "words": []}, {"text": "b", "words": []}]
        kept, dropped = filter_empty_segments(segs)
        assert len(kept) == 2
        assert dropped == 0

    def test_keeps_segments_with_words_but_no_text(self) -> None:
        segs = [{"text": "", "text_ITN": "", "words": [{"word": "hi"}]}]
        kept, dropped = filter_empty_segments(segs)
        assert len(kept) == 1
        assert dropped == 0

    def test_drops_fully_empty_segments(self) -> None:
        segs = [
            {"text": "a", "words": []},
            {"text": "", "text_ITN": "", "words": []},
            {"text": "b", "words": []},
        ]
        kept, dropped = filter_empty_segments(segs)
        assert [s["text"] for s in kept] == ["a", "b"]
        assert dropped == 1

    def test_preserves_input_order(self) -> None:
        segs = [{"text": str(i), "words": []} for i in range(5)]
        kept, _ = filter_empty_segments(segs)
        assert [s["text"] for s in kept] == ["0", "1", "2", "3", "4"]


# ----------------------------------------------------------------------
# find_overlapping_indices
# ----------------------------------------------------------------------


def _seg(start: float, end: float) -> dict:
    return {"start": start, "end": end}


class TestFindOverlappingIndices:
    def test_no_overlaps(self) -> None:
        segs = [_seg(0, 1), _seg(2, 3), _seg(4, 5)]
        assert find_overlapping_indices(segs, 0.5) == set()

    def test_large_overlap_flags_both(self) -> None:
        # 1.0s overlap, neither contains the other → both flagged
        segs = [_seg(0, 5), _seg(4, 8)]
        assert find_overlapping_indices(segs, 0.5) == {0, 1}

    def test_small_overlap_kept(self) -> None:
        # 0.3s overlap, < min_overlap, no containment → not flagged
        segs = [_seg(0, 5), _seg(4.7, 8)]
        assert find_overlapping_indices(segs, 0.5) == set()

    def test_small_overlap_with_full_containment_flagged(self) -> None:
        # Even 0.1s overlap is flagged if one contains the other.
        segs = [_seg(0, 10), _seg(4, 4.1)]
        assert find_overlapping_indices(segs, 0.5) == {0, 1}

    def test_chain_of_overlaps_flags_all(self) -> None:
        # A overlaps B and B overlaps C; A and C don't overlap.
        # Overlap relation is computed pairwise — A, B, C all flagged.
        segs = [_seg(0, 5), _seg(4, 9), _seg(8, 12)]
        assert find_overlapping_indices(segs, 0.5) == {0, 1, 2}

    def test_min_overlap_boundary_inclusive(self) -> None:
        # Exactly 0.5s overlap → flagged ( >= )
        segs = [_seg(0, 5), _seg(4.5, 8)]
        assert find_overlapping_indices(segs, 0.5) == {0, 1}

    def test_unsorted_input_still_correct(self) -> None:
        segs = [_seg(10, 12), _seg(0, 5), _seg(4, 8)]
        # Indices 1 and 2 overlap (1.0s); index 0 is fine
        assert find_overlapping_indices(segs, 0.5) == {1, 2}

    def test_empty_input(self) -> None:
        assert find_overlapping_indices([], 0.5) == set()

    def test_single_segment(self) -> None:
        assert find_overlapping_indices([_seg(0, 5)], 0.5) == set()

    def test_identical_intervals(self) -> None:
        # Sweep-line + heap edge case: equal start AND end. Both should
        # be flagged via containment (each "contains" the other).
        segs = [_seg(0, 5), _seg(0, 5)]
        assert find_overlapping_indices(segs, 0.5) == {0, 1}

    def test_long_non_overlapping_chain(self) -> None:
        # 200 strictly non-overlapping intervals. With the old O(n^2) pair
        # scan this is ~20k comparisons; with the sweep-line each new
        # interval finds the heap empty after eviction, so the inner loop
        # is never entered. Asserts the result is empty regardless.
        segs = [_seg(i * 2.0, i * 2.0 + 1.0) for i in range(200)]
        assert find_overlapping_indices(segs, 0.5) == set()

    def test_two_speakers_alternating_overlap(self) -> None:
        # Simulate two interleaved speakers (A: 0-5, B: 4-9, A: 8-13,
        # B: 12-17). Adjacent pairs overlap by 1.0s; far pairs don't.
        # Sweep-line must evict the trailing A/B from the heap as the
        # cursor moves forward so the active set stays bounded.
        segs = [_seg(0, 5), _seg(4, 9), _seg(8, 13), _seg(12, 17)]
        assert find_overlapping_indices(segs, 0.5) == {0, 1, 2, 3}


# ----------------------------------------------------------------------
# plan_snippets
# ----------------------------------------------------------------------


def _ts(start: float, end: float, text: str = "x") -> dict:
    return {"start": start, "end": end, "text": text, "text_ITN": text, "words": []}


class TestPlanSnippets:
    def test_empty_input(self) -> None:
        snippets, drops = plan_snippets([], 30.0, 0.5, 30.0)
        assert snippets == []
        assert drops == {"too_long": 0, "too_short": 0, "no_text": 0}

    def test_single_short_segment(self) -> None:
        snippets, _drops = plan_snippets([_ts(0, 5)], 30.0, 0.5, 30.0)
        assert len(snippets) == 1
        assert (snippets[0]["start"], snippets[0]["end"]) == (0, 5)

    def test_too_long_dropped(self) -> None:
        snippets, drops = plan_snippets([_ts(0, 50)], 30.0, 0.5, 30.0)
        assert snippets == []
        assert drops["too_long"] == 1

    def test_too_short_dropped(self) -> None:
        snippets, drops = plan_snippets([_ts(0, 0.2)], 30.0, 0.5, 30.0)
        assert snippets == []
        assert drops["too_short"] == 1

    def test_no_text_dropped(self) -> None:
        seg = {"start": 0, "end": 5, "text": "", "text_ITN": "", "words": []}
        snippets, drops = plan_snippets([seg], 30.0, 0.5, 30.0)
        assert snippets == []
        assert drops["no_text"] == 1

    def test_greedy_packing_within_max_duration(self) -> None:
        segs = [_ts(0, 5), _ts(5, 10), _ts(10, 15), _ts(15, 20)]
        snippets, _ = plan_snippets(segs, max_duration_sec=20.0, min_duration_sec=0.5, max_segment_gap_in_snippet=30.0)
        # All four pack into one snippet of [0, 20]
        assert len(snippets) == 1
        assert (snippets[0]["start"], snippets[0]["end"]) == (0, 20)
        assert len(snippets[0]["segments"]) == 4

    def test_max_duration_forces_split(self) -> None:
        segs = [_ts(0, 8), _ts(8, 14), _ts(14, 20)]
        # max=15: 0-14 fits (14s), but +20 would make 20s → splits at the third
        snippets, _ = plan_snippets(segs, 15.0, 0.5, 30.0)
        assert [(s["start"], s["end"]) for s in snippets] == [(0, 14), (14, 20)]

    def test_gap_within_threshold_kept(self) -> None:
        segs = [_ts(0, 3), _ts(5, 8)]  # 2s gap
        snippets, _ = plan_snippets(segs, 30.0, 0.5, max_segment_gap_in_snippet=2.0)
        assert len(snippets) == 1
        assert (snippets[0]["start"], snippets[0]["end"]) == (0, 8)

    def test_gap_above_threshold_forces_split(self) -> None:
        segs = [_ts(0, 3), _ts(5.5, 8)]  # 2.5s gap
        snippets, _ = plan_snippets(segs, 30.0, 0.5, max_segment_gap_in_snippet=2.0)
        assert [(s["start"], s["end"]) for s in snippets] == [(0, 3), (5.5, 8)]

    def test_gap_zero_groups_only_contiguous(self) -> None:
        segs = [_ts(0, 3), _ts(3, 5), _ts(5.1, 7)]
        snippets, _ = plan_snippets(segs, 30.0, 0.5, max_segment_gap_in_snippet=0.0)
        # First two are contiguous (gap 0), third has 0.1s gap → split
        assert len(snippets) == 2
        assert (snippets[0]["start"], snippets[0]["end"]) == (0, 5)
        assert (snippets[1]["start"], snippets[1]["end"]) == (5.1, 7)

    def test_combined_drops(self) -> None:
        segs = [
            _ts(0, 5),  # ok
            _ts(10, 60),  # too_long (50s)
            _ts(70, 70.1),  # too_short
            {"start": 100, "end": 105, "text": "", "text_ITN": "", "words": []},  # no_text
        ]
        snippets, drops = plan_snippets(segs, 30.0, 0.5, 30.0)
        assert len(snippets) == 1
        assert drops == {"too_long": 1, "too_short": 1, "no_text": 1}


# ----------------------------------------------------------------------
# relativize_segments
# ----------------------------------------------------------------------


class TestRelativizeSegments:
    def test_shifts_segment_timestamps(self) -> None:
        segs = [{"start": 10.0, "end": 12.0, "text": "t"}]
        out = relativize_segments(segs, snippet_start=10.0, snippet_end=12.0)
        assert out[0]["start"] == 0.0
        assert out[0]["end"] == 2.0

    def test_shifts_word_timestamps(self) -> None:
        segs = [
            {
                "start": 10.0,
                "end": 12.0,
                "text": "t",
                "words": [{"word": "hi", "start": 10.5, "end": 11.0}],
            }
        ]
        out = relativize_segments(segs, 10.0, 12.0)
        assert out[0]["words"][0]["start"] == pytest.approx(0.5)
        assert out[0]["words"][0]["end"] == pytest.approx(1.0)

    def test_missing_words_key_ok(self) -> None:
        segs = [{"start": 5.0, "end": 6.0, "text": "t"}]
        out = relativize_segments(segs, 5.0, 6.0)
        assert "words" not in out[0]

    def test_does_not_mutate_input(self) -> None:
        segs = [{"start": 10.0, "end": 12.0, "text": "t", "words": [{"word": "x", "start": 10.5, "end": 11.0}]}]
        relativize_segments(segs, 10.0, 12.0)
        assert segs[0]["start"] == 10.0
        assert segs[0]["words"][0]["start"] == 10.5

    def test_preserves_other_fields(self) -> None:
        segs = [{"start": 10.0, "end": 12.0, "text": "t", "speaker": "A", "extra": 42}]
        out = relativize_segments(segs, 10.0, 12.0)
        assert out[0]["speaker"] == "A"
        assert out[0]["extra"] == 42

    def test_clamps_negative_to_zero(self) -> None:
        # Word annotated as starting slightly before its parent segment
        # (real-world data jitter).  After shifting by snippet_start, the
        # raw value would be negative; clamping pulls it to 0.
        segs = [
            {
                "start": 10.0,
                "end": 12.0,
                "text": "t",
                "words": [{"word": "hi", "start": 9.95, "end": 11.0}],
            }
        ]
        out = relativize_segments(segs, snippet_start=10.0, snippet_end=12.0)
        assert out[0]["words"][0]["start"] == 0.0
        assert out[0]["words"][0]["end"] == pytest.approx(1.0)

    def test_clamps_above_duration_to_duration(self) -> None:
        segs = [
            {
                "start": 10.0,
                "end": 12.05,  # 0.05s past snippet end
                "text": "t",
                "words": [{"word": "x", "start": 11.5, "end": 12.5}],  # word ends past
            }
        ]
        out = relativize_segments(segs, snippet_start=10.0, snippet_end=12.0)
        assert out[0]["end"] == pytest.approx(2.0)
        assert out[0]["words"][0]["end"] == pytest.approx(2.0)


# ----------------------------------------------------------------------
# make_snippet_id
# ----------------------------------------------------------------------


class TestMakeSnippetId:
    def test_three_decimal_format(self) -> None:
        assert make_snippet_id("XYZ", 11.708468, 13.969718) == "XYZ-11_708-13_970"

    def test_zero_padded_decimals(self) -> None:
        assert make_snippet_id("X", 1.0, 2.5) == "X-1_000-2_500"

    def test_distinct_for_close_timestamps(self) -> None:
        # Two snippets that would collide at .2f precision are distinct at .3f
        assert make_snippet_id("X", 12.123, 13.0) != make_snippet_id("X", 12.127, 13.0)

    def test_no_dot_in_id_for_webdataset(self) -> None:
        # Filenames are derived as `{snippet_id}.<ext>`. WebDataset uses the
        # first `.` after the sample basename as the boundary between key
        # and extension(s); any `.` inside the snippet_id would cause it to
        # be misparsed as a multi-piece compound key. Guard the contract here.
        sid = make_snippet_id("source-id_with-dashes", 3101.250, 3109.940)
        assert "." not in sid
        assert sid == "source-id_with-dashes-3101_250-3109_940"

    def test_dots_in_original_id_are_sanitized(self) -> None:
        # Input manifests sometimes carry an id like a source filename
        # ("meeting.wav") or a versioned id ("session.1.2"). Those dots
        # would otherwise survive into the snippet id and break the same
        # WebDataset contract as dotted timestamps.
        sid = make_snippet_id("meeting.wav", 0.0, 30.0)
        assert "." not in sid
        assert sid == "meeting_wav-0_000-30_000"

        sid = make_snippet_id("session.1.2", 1.0, 2.5)
        assert "." not in sid
        assert sid == "session_1_2-1_000-2_500"

    def test_path_separators_in_original_id_are_sanitized(self) -> None:
        # Path-like ids would otherwise turn `<snippet_id>.<ext>` into a
        # nested tar path, breaking the "members live at the tar root"
        # contract.  Both POSIX and Windows separators are stripped so the
        # output stays portable.
        sid = make_snippet_id("shard1/utt001", 0.0, 5.0)
        assert "/" not in sid
        assert "\\" not in sid
        assert sid == "shard1_utt001-0_000-5_000"

        sid = make_snippet_id("dirA\\sub\\utt", 1.0, 2.0)
        assert "/" not in sid
        assert "\\" not in sid
        assert sid == "dirA_sub_utt-1_000-2_000"


# ----------------------------------------------------------------------
# histogram_30s
# ----------------------------------------------------------------------


class TestHistogram30s:
    def test_empty_input(self) -> None:
        assert histogram_30s([]) == {}

    def test_single_bucket(self) -> None:
        assert histogram_30s([5.0, 12.0, 29.999]) == {"0-30": 3}

    def test_multi_bucket_ordered(self) -> None:
        out = histogram_30s([5.0, 35.0, 65.0])
        assert list(out.keys()) == ["0-30", "30-60", "60-90"]
        assert out == {"0-30": 1, "30-60": 1, "60-90": 1}

    def test_30_lands_in_30_60_bucket(self) -> None:
        # 30 // 30 == 1 → second bin
        assert histogram_30s([30.0]) == {"0-30": 0, "30-60": 1}


# ----------------------------------------------------------------------
# _resolve_audio_path
# ----------------------------------------------------------------------


class TestResolveAudioPath:
    def test_relative_path_basename(self) -> None:
        assert _resolve_audio_path("/data", "./foo.wav") == "/data/foo.wav"

    def test_absolute_path_uses_basename(self) -> None:
        assert _resolve_audio_path("/data", "/elsewhere/bar.m4a") == "/data/bar.m4a"

    def test_simple_basename(self) -> None:
        assert _resolve_audio_path("/data", "baz.flac") == "/data/baz.flac"

    def test_relative_mode_preserves_subdirectories(self) -> None:
        # 'relative' joins audio_dir / value verbatim, so a manifest that
        # carries a relative sub-path stays under that sub-path on disk.
        assert (
            _resolve_audio_path("/data", "shard1/foo.wav", "relative") == "/data/shard1/foo.wav"
        )

    def test_relative_mode_absolute_value_wins(self) -> None:
        # os.path.join semantics: an absolute right-hand side discards the
        # left-hand side. Documented in the docstring as the way 'relative'
        # mode handles absolute manifest paths.
        assert (
            _resolve_audio_path("/data", "/elsewhere/bar.wav", "relative") == "/elsewhere/bar.wav"
        )

    def test_as_is_mode_returns_value_unchanged(self) -> None:
        assert (
            _resolve_audio_path("/data", "/elsewhere/bar.wav", "as_is") == "/elsewhere/bar.wav"
        )
        assert _resolve_audio_path("/data", "rel/foo.wav", "as_is") == "rel/foo.wav"

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown audio_path_resolution"):
            _resolve_audio_path("/data", "foo.wav", "not_a_mode")


# ----------------------------------------------------------------------
# Shard helpers
# ----------------------------------------------------------------------


class TestShardHelpers:
    def test_shard_path_unique(self, tmp_path: Path) -> None:
        out = str(tmp_path / "manifest.jsonl")
        a = _make_shard_path(out, "jsonl")
        b = _make_shard_path(out, "jsonl")
        assert a != b
        assert a.startswith(out + ".shard-")
        assert a.endswith(".jsonl")

    def test_glob_shards_finds_only_matching(self, tmp_path: Path) -> None:
        out = str(tmp_path / "manifest.jsonl")
        s1 = _make_shard_path(out, "jsonl")
        s2 = _make_shard_path(out, "jsonl")
        unrelated = str(tmp_path / "other.jsonl")
        for p in (s1, s2, unrelated):
            Path(p).touch()
        found = _glob_shards(out, "jsonl")
        assert sorted(found) == sorted([s1, s2])

    def test_delete_shards_removes_and_returns_count(self, tmp_path: Path) -> None:
        out = str(tmp_path / "manifest.jsonl")
        for _ in range(3):
            Path(_make_shard_path(out, "jsonl")).touch()
        n = _delete_shards(out, "jsonl")
        assert n == 3
        assert _glob_shards(out, "jsonl") == []


# ----------------------------------------------------------------------
# _build_final_summary
# ----------------------------------------------------------------------


class TestBuildFinalSummary:
    def test_empty(self) -> None:
        s = _build_final_summary({}, [])
        assert s["num_input_audios"] == 0
        assert s["num_output_snippets"] == 0
        assert s["snippet_duration_histogram_30s"] == {}
        assert s["per_original"] == []

    def test_aggregates_totals_and_dropped(self) -> None:
        per_original = {
            "a": {
                "id": "a",
                "in_segments": 10,
                "in_duration_sec": 100.0,
                "dropped": {"empty": 1, "overlap": 2, "too_long": 0, "too_short": 1, "no_text": 0},
                "out_snippets": 3,
                "out_segments": 7,
                "out_duration_sec": 25.0,
            },
            "b": {
                "id": "b",
                "in_segments": 5,
                "in_duration_sec": 50.0,
                "dropped": {"empty": 0, "overlap": 0, "too_long": 1, "too_short": 0, "no_text": 0},
                "out_snippets": 1,
                "out_segments": 2,
                "out_duration_sec": 10.0,
            },
        }
        s = _build_final_summary(per_original, [25.0, 10.0])
        assert s["num_input_audios"] == 2
        assert s["num_output_snippets"] == 4
        assert s["input_total_segments"] == 15
        assert s["input_total_duration_sec"] == 150.0
        assert s["output_total_segments"] == 9
        assert s["output_total_duration_sec"] == 35.0
        assert s["dropped"] == {"empty": 1, "overlap": 2, "too_long": 1, "too_short": 1, "no_text": 0}
        assert s["snippet_duration_histogram_30s"] == {"0-30": 2}
        assert len(s["per_original"]) == 2


# ----------------------------------------------------------------------
# Repetition-filter helpers
# ----------------------------------------------------------------------


class TestCountNgrams:
    def test_counts_overlapping_windows(self) -> None:
        # [1,2,3] repeated three times -> 3-gram (1,2,3) appears 3 times
        c = _count_ngrams([1, 2, 3, 1, 2, 3, 1, 2, 3], 3)
        assert c[(1, 2, 3)] == 3
        # 2-gram across the boundary: (3,1) appears twice
        c2 = _count_ngrams([1, 2, 3, 1, 2, 3, 1, 2, 3], 2)
        assert c2[(1, 2)] == 3
        assert c2[(3, 1)] == 2

    def test_returns_empty_when_too_short(self) -> None:
        assert _count_ngrams([1, 2], 4) == Counter()
        assert _count_ngrams([], 2) == Counter()

    def test_returns_empty_when_n_invalid(self) -> None:
        assert _count_ngrams([1, 2, 3], 0) == Counter()
        assert _count_ngrams([1, 2, 3], -1) == Counter()


class TestFindOffendingNgrams:
    def test_strict_inequality(self) -> None:
        c = Counter({(1, 2): 3, (3, 4): 4, (5, 6): 5})
        # max_count=3 -> only counts > 3 are offending
        assert _find_offending_ngrams(c, 3) == {(3, 4), (5, 6)}

    def test_empty_when_none_exceed(self) -> None:
        c = Counter({(1, 2): 3, (3, 4): 1})
        assert _find_offending_ngrams(c, 3) == set()


class TestLocateNgramCharRanges:
    def test_locates_every_occurrence(self) -> None:
        # Three repetitions of 3-gram (1,2,3) at non-touching offsets
        toks = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        offs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)]
        ranges = _locate_ngram_char_ranges(toks, offs, {(1, 2, 3)}, 3)
        assert ranges == [(0, 5), (6, 11), (12, 17)]

    def test_empty_when_no_offending(self) -> None:
        assert _locate_ngram_char_ranges([1, 2, 3], [(0, 1), (2, 3), (4, 5)], set(), 2) == []

    def test_empty_when_too_short(self) -> None:
        assert _locate_ngram_char_ranges([1], [(0, 1)], {(1, 2)}, 2) == []


class TestMergeCharRanges:
    def test_merges_overlapping(self) -> None:
        assert _merge_char_ranges([(0, 5), (3, 8), (10, 15)]) == [(0, 8), (10, 15)]

    def test_sorts_input(self) -> None:
        assert _merge_char_ranges([(10, 15), (0, 5)]) == [(0, 5), (10, 15)]

    def test_merges_touching(self) -> None:
        assert _merge_char_ranges([(0, 5), (5, 10)]) == [(0, 10)]

    def test_empty(self) -> None:
        assert _merge_char_ranges([]) == []


class TestFormatRed:
    def test_wraps_single_range(self) -> None:
        assert _format_red("hello world", [(0, 5)]) == "<red>hello</red> world"

    def test_no_ranges_returns_text(self) -> None:
        assert _format_red("hello world", []) == "hello world"

    def test_escapes_lt_in_plain_text(self) -> None:
        assert _format_red("a<b>c", []) == r"a\<b>c"

    def test_escapes_lt_inside_highlighted(self) -> None:
        # "<" inside a highlighted span must be escaped too so loguru doesn't try to parse it.
        assert _format_red("<x>", [(0, 3)]) == r"<red>\<x></red>"

    def test_multiple_ranges(self) -> None:
        assert _format_red("aaa bbb ccc", [(0, 3), (8, 11)]) == "<red>aaa</red> bbb <red>ccc</red>"
