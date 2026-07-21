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

"""Unit tests for bench_msrvtt_retrieval.py helper functions (CPU only)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from eval.video.bench_msrvtt_retrieval import (
    _build_video_index,
    _normalise,
    build_corpus,
    compute_metrics,
    embed_texts,
    embed_videos,
    parse_args,
    rank_queries,
    t2v_gt_ranks,
    v2t_gt_ranks,
)


# ---- _normalise ----


class TestNormalise:
    def test_unit_vectors_unchanged(self) -> None:
        mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        out = _normalise(mat)
        np.testing.assert_allclose(out, mat, atol=1e-6)

    def test_norms_become_one(self) -> None:
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((8, 16)).astype(np.float32)
        out = _normalise(mat)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(8), atol=1e-5)

    def test_zero_row_not_nan(self) -> None:
        mat = np.zeros((3, 4), dtype=np.float32)
        out = _normalise(mat)
        assert not np.any(np.isnan(out))

    def test_scalar_multiple_same_direction(self) -> None:
        mat = np.array([[3.0, 4.0]], dtype=np.float32)
        out = _normalise(mat)
        np.testing.assert_allclose(out, [[0.6, 0.8]], atol=1e-6)


# ---- build_corpus ----


class TestBuildCorpus:
    def test_deduplicates_videos(self) -> None:
        records = [
            {"video_id": "vid1", "video_path": "/a/vid1.mp4", "caption": "c1"},
            {"video_id": "vid1", "video_path": "/a/vid1.mp4", "caption": "c2"},
            {"video_id": "vid2", "video_path": "/a/vid2.mp4", "caption": "c3"},
        ]
        ids, paths = build_corpus(records)
        assert len(ids) == 2
        assert len(paths) == 2

    def test_preserves_insertion_order(self) -> None:
        records = [
            {"video_id": "b", "video_path": "/b.mp4", "caption": "x"},
            {"video_id": "a", "video_path": "/a.mp4", "caption": "y"},
        ]
        ids, paths = build_corpus(records)
        assert ids == ["b", "a"]
        assert paths == ["/b.mp4", "/a.mp4"]

    def test_first_path_wins_for_duplicate(self) -> None:
        records = [
            {"video_id": "vid1", "video_path": "/first.mp4", "caption": "a"},
            {"video_id": "vid1", "video_path": "/second.mp4", "caption": "b"},
        ]
        ids, paths = build_corpus(records)
        assert paths == ["/first.mp4"]

    def test_empty_records(self) -> None:
        ids, paths = build_corpus([])
        assert ids == []
        assert paths == []


# ---- rank_queries ----


class TestRankQueries:
    def test_perfect_match_ranks_first(self) -> None:
        corpus = np.eye(4, dtype=np.float32)
        queries = np.eye(4, dtype=np.float32)
        ranks = rank_queries(queries, corpus)
        for i in range(4):
            assert ranks[i, i] == 1

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(1)
        q = rng.standard_normal((5, 8)).astype(np.float32)
        c = rng.standard_normal((10, 8)).astype(np.float32)
        ranks = rank_queries(q, c)
        assert ranks.shape == (5, 10)

    def test_ranks_are_1_based(self) -> None:
        rng = np.random.default_rng(2)
        q = rng.standard_normal((3, 4)).astype(np.float32)
        c = rng.standard_normal((6, 4)).astype(np.float32)
        ranks = rank_queries(q, c)
        assert ranks.min() == 1
        assert ranks.max() == 6

    def test_each_row_is_permutation(self) -> None:
        rng = np.random.default_rng(3)
        q = rng.standard_normal((4, 8)).astype(np.float32)
        c = rng.standard_normal((7, 8)).astype(np.float32)
        ranks = rank_queries(q, c)
        for row in ranks:
            assert sorted(row) == list(range(1, 8))


# ---- compute_metrics ----


class TestComputeMetrics:
    def test_perfect_retrieval(self) -> None:
        ranks = np.ones(10, dtype=np.float32)
        m = compute_metrics(ranks, top_k=10)
        assert m["recall_at_1"] == pytest.approx(1.0)
        assert m["recall_at_5"] == pytest.approx(1.0)
        assert m["recall_at_10"] == pytest.approx(1.0)
        assert m["median_rank"] == pytest.approx(1.0)
        assert m["mean_rank"] == pytest.approx(1.0)
        assert m["mrr"] == pytest.approx(1.0)

    def test_all_at_last_rank(self) -> None:
        n = 100
        ranks = np.full(n, n, dtype=np.float32)
        m = compute_metrics(ranks, top_k=10)
        assert m["recall_at_1"] == pytest.approx(0.0)
        assert m["recall_at_5"] == pytest.approx(0.0)
        assert m["recall_at_10"] == pytest.approx(0.0)

    def test_total_queries_field(self) -> None:
        ranks = np.arange(1, 21, dtype=np.float32)
        m = compute_metrics(ranks, top_k=10)
        assert m["total_queries"] == 20

    def test_custom_top_k_key(self) -> None:
        ranks = np.ones(5, dtype=np.float32)
        m = compute_metrics(ranks, top_k=7)
        assert "recall_at_7" in m
        assert "ndcg_at_7" in m

    def test_mrr_half(self) -> None:
        ranks = np.array([2.0, 2.0], dtype=np.float32)
        m = compute_metrics(ranks, top_k=10)
        assert m["mrr"] == pytest.approx(0.5)

    def test_corpus_size_initially_none(self) -> None:
        ranks = np.ones(3, dtype=np.float32)
        m = compute_metrics(ranks, top_k=10)
        assert m["corpus_size"] is None


# ---- t2v_gt_ranks ----


class TestT2VGtRanks:
    def _identity(self, n: int) -> np.ndarray:
        return np.eye(n, dtype=np.float32)

    def test_perfect_alignment(self) -> None:
        n = 4
        vid_ids = [f"v{i}" for i in range(n)]
        vid_to_idx = {v: i for i, v in enumerate(vid_ids)}
        ranks = t2v_gt_ranks(self._identity(n), self._identity(n), vid_ids, vid_to_idx)
        np.testing.assert_array_equal(ranks, np.ones(n))

    def test_missing_video_gets_worst_rank(self) -> None:
        n = 3
        query_gt = ["v0", "v1", "MISSING"]
        vid_to_idx = {"v0": 0, "v1": 1, "v2": 2}
        ranks = t2v_gt_ranks(self._identity(n), self._identity(n), query_gt, vid_to_idx)
        assert ranks[2] == n + 1

    def test_output_length_matches_queries(self) -> None:
        rng = np.random.default_rng(10)
        q, v = 5, 8
        text_embs = rng.standard_normal((q, 16)).astype(np.float32)
        vid_embs = rng.standard_normal((v, 16)).astype(np.float32)
        gt = [f"v{i % v}" for i in range(q)]
        vid_to_idx = {f"v{i}": i for i in range(v)}
        ranks = t2v_gt_ranks(text_embs, vid_embs, gt, vid_to_idx)
        assert ranks.shape == (q,)


# ---- v2t_gt_ranks ----


class TestV2TGtRanks:
    def _identity(self, n: int) -> np.ndarray:
        return np.eye(n, dtype=np.float32)

    def test_perfect_alignment(self) -> None:
        n = 4
        corpus_vids = [f"v{i}" for i in range(n)]
        query_gt = [f"v{i}" for i in range(n)]
        ranks = v2t_gt_ranks(self._identity(n), self._identity(n), corpus_vids, query_gt)
        np.testing.assert_array_equal(ranks, np.ones(n))

    def test_multiple_captions_per_video_uses_best(self) -> None:
        # vid0 matches text0 perfectly; vid1 has two captions (text1 closer than text2)
        video_embs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        text_embs = np.array([[1, 0, 0], [0, 1, 0], [0, 0.5, 0.5]], dtype=np.float32)
        corpus_vids = ["v0", "v1"]
        query_gt = ["v0", "v1", "v1"]
        ranks = v2t_gt_ranks(video_embs, text_embs, corpus_vids, query_gt)
        assert ranks[0] == 1
        assert ranks[1] == 1

    def test_output_length_matches_corpus(self) -> None:
        rng = np.random.default_rng(20)
        v, q = 6, 10
        vid_embs = rng.standard_normal((v, 12)).astype(np.float32)
        txt_embs = rng.standard_normal((q, 12)).astype(np.float32)
        corpus_vids = [f"v{i}" for i in range(v)]
        query_gt = [f"v{i % v}" for i in range(q)]
        ranks = v2t_gt_ranks(vid_embs, txt_embs, corpus_vids, query_gt)
        assert ranks.shape == (v,)

    def test_video_with_no_captions_gets_worst_rank(self) -> None:
        video_embs = np.eye(2, dtype=np.float32)
        text_embs = np.eye(2, dtype=np.float32)
        corpus_vids = ["v0", "v1"]
        query_gt = ["v0", "v0"]  # v1 has no caption
        ranks = v2t_gt_ranks(video_embs, text_embs, corpus_vids, query_gt)
        assert ranks[1] == len(query_gt) + 1


# ---- _build_video_index ----


class TestBuildVideoIndex:
    def test_indexes_mp4_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "vid0.mp4").touch()
            (root / "vid1.mp4").touch()
            index = _build_video_index(root)
            assert "vid0" in index
            assert "vid1" in index
            assert len(index) == 2

    def test_ignores_non_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "clip.avi").touch()
            (root / "clip.mp4").touch()
            index = _build_video_index(root)
            assert "clip" in index
            assert len(index) == 1

    def test_recursive_search(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sub = root / "subdir"
            sub.mkdir()
            (sub / "nested.mp4").touch()
            index = _build_video_index(root)
            assert "nested" in index

    def test_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            index = _build_video_index(Path(d))
            assert index == {}


# ---- parse_args ----


class TestParseArgs:
    def test_required_model_dir(self) -> None:
        args = parse_args(["--model-dir", "/path/to/models"])
        assert args.model_dir == "/path/to/models"

    def test_defaults(self) -> None:
        args = parse_args(["--model-dir", "/models"])
        assert args.variant == "336p"
        assert args.split == "test"
        assert args.top_k == 10
        assert args.limit == 0
        assert args.sample_fps == pytest.approx(2.0)
        assert args.text_batch_size == 64
        assert args.video_batch_size == 8
        assert args.direction == "both"
        assert args.output is None

    def test_custom_values(self) -> None:
        args = parse_args([
            "--model-dir", "/models",
            "--variant", "224p",
            "--split", "val",
            "--top-k", "5",
            "--limit", "100",
            "--sample-fps", "4.0",
            "--direction", "t2v",
        ])
        assert args.variant == "224p"
        assert args.split == "val"
        assert args.top_k == 5
        assert args.limit == 100
        assert args.sample_fps == pytest.approx(4.0)
        assert args.direction == "t2v"

    def test_missing_model_dir_raises(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([])


# ---- embed_texts ----

_DIM = 16
_FAKE_EMB = torch.zeros(1, _DIM, dtype=torch.float16)


def _text_model(dim: int = _DIM) -> MagicMock:
    m = MagicMock()
    m.get_text_embedding.return_value = torch.randn(1, dim, dtype=torch.float16)
    return m


class TestEmbedTexts:
    def test_output_shape(self) -> None:
        out = embed_texts(_text_model(), ["a", "b", "c"], batch_size=2)
        assert out.shape == (3, _DIM)
        assert out.dtype == np.float32

    def test_output_is_normalised(self) -> None:
        model = MagicMock()
        model.get_text_embedding.return_value = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
        out = embed_texts(model, ["x", "y"], batch_size=1)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(2), atol=1e-5)

    def test_calls_model_once_per_text(self) -> None:
        model = _text_model()
        embed_texts(model, ["a", "b", "c", "d"], batch_size=64)
        assert model.get_text_embedding.call_count == 4

    def test_single_text(self) -> None:
        out = embed_texts(_text_model(), ["hello"], batch_size=1)
        assert out.shape == (1, _DIM)


# ---- embed_videos ----

_FAKE_FRAMES = [np.zeros((4, 4, 3), dtype=np.uint8)]


def _video_model(dim: int = _DIM) -> MagicMock:
    m = MagicMock()
    m.formulate_input_frames.return_value = object()
    m.encode_video_frames.return_value = torch.randn(1, dim, dtype=torch.float16)
    return m


class TestEmbedVideos:
    def test_successful_embed_sets_valid_true(self) -> None:
        with patch("eval.video.bench_msrvtt_retrieval._load_frames", return_value=_FAKE_FRAMES):
            embs, valid = embed_videos(_video_model(), ["/v.mp4"], batch_size=1)
        assert valid[0]
        assert embs.shape == (1, _DIM)

    def test_failed_video_sets_valid_false(self) -> None:
        def _loader(path, fps):
            if "good" in path:
                return _FAKE_FRAMES
            raise ValueError("undecodable")

        with patch("eval.video.bench_msrvtt_retrieval._load_frames", side_effect=_loader):
            embs, valid = embed_videos(_video_model(), ["/bad.mp4", "/good.mp4"], batch_size=1)
        assert not valid[0]
        assert valid[1]

    def test_output_shape_and_dtype(self) -> None:
        with patch("eval.video.bench_msrvtt_retrieval._load_frames", return_value=_FAKE_FRAMES):
            embs, valid = embed_videos(_video_model(), ["/a.mp4", "/b.mp4"], batch_size=1)
        assert embs.shape == (2, _DIM)
        assert embs.dtype == np.float32
        assert valid.shape == (2,)
        assert valid.dtype == bool

    def test_all_failed_raises(self) -> None:
        with patch("eval.video.bench_msrvtt_retrieval._load_frames", side_effect=ValueError("bad")):
            with pytest.raises(RuntimeError, match="No videos could be decoded"):
                embed_videos(_video_model(), ["/bad.mp4"], batch_size=1)

    def test_zero_embeddings_for_failed_are_not_nan(self) -> None:
        model = _video_model()

        def _loader(path, fps):
            if "good" in path:
                return _FAKE_FRAMES
            raise ValueError("bad")

        with patch("eval.video.bench_msrvtt_retrieval._load_frames", side_effect=_loader):
            embs, valid = embed_videos(model, ["/bad.mp4", "/good.mp4"], batch_size=1)
        assert not np.any(np.isnan(embs))
