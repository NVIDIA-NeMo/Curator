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

"""Unit tests for caption_clipscore.py helper functions (CPU only)."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

import numpy as np
import pytest
import torch

from eval.video.caption_clipscore import (
    _collect_tasks,
    _cosine_sim,
    _get_source_video,
    _get_window_captions,
    _load_uid_list,
    _load_video_embeddings,
)


@pytest.fixture
def tmp_dir() -> Generator[str]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_meta(tmp_dir: str) -> str:
    """Create a sample clip metadata JSON."""
    meta = {
        "source_video": "/data/video.mp4",
        "duration_span": [0.0, 10.0],
        "windows": [
            {"qwen2.5_caption": "A dog runs across a field."},
            {"qwen2.5_caption": "The dog jumps over a fence."},
        ],
    }
    path = os.path.join(tmp_dir, "clip001.json")
    with open(path, "w") as f:
        json.dump(meta, f)
    return path


@pytest.fixture
def embedding_dir(tmp_dir: str) -> str:
    """Create a directory with fake pickle embeddings."""
    emb_dir = os.path.join(tmp_dir, "ce1_embd")
    os.makedirs(emb_dir)
    rng = np.random.default_rng(42)
    for uid in ["uid-aaa", "uid-bbb", "uid-ccc"]:
        arr = rng.standard_normal((1, 256)).astype(np.float32)
        with open(os.path.join(emb_dir, f"{uid}.pickle"), "wb") as f:
            pickle.dump(arr, f)
    return emb_dir


@pytest.fixture
def caption_dirs(tmp_dir: str) -> dict[str, str]:
    """Create caption directories with metadata."""
    dirs: dict[str, str] = {}
    for label in ["qwen25", "nemotron"]:
        cap_dir = os.path.join(tmp_dir, f"captions_{label}")
        meta_dir = os.path.join(cap_dir, "metas", "v0")
        os.makedirs(meta_dir)
        for uid in ["uid-aaa", "uid-bbb", "uid-ccc"]:
            meta = {
                "source_video": f"/data/{uid}_source.mp4",
                "duration_span": [0.0, 10.0],
                "windows": [{f"{label}_caption": f"Caption for {uid} by {label}."}],
            }
            with open(os.path.join(meta_dir, f"{uid}.json"), "w") as f:
                json.dump(meta, f)
        dirs[label] = cap_dir
    return dirs


@pytest.fixture
def uid_list_file(tmp_dir: str) -> str:
    """Create a simple UID list file."""
    path = os.path.join(tmp_dir, "selected_uids.txt")
    with open(path, "w") as f:
        f.write("uid-aaa\nuid-bbb\n")
    return path


@pytest.fixture
def uid_list_file_with_spans(tmp_dir: str) -> str:
    """Create a UID list with tab-separated source_video and spans."""
    path = os.path.join(tmp_dir, "selected_uids_spans.txt")
    with open(path, "w") as f:
        f.write("uid-xxx\tuid-aaa_source.mp4\t0.0\t10.0\n")
        f.write("uid-yyy\tuid-bbb_source.mp4\t0.0\t10.0\n")
    return path


# ---- _get_window_captions ----


class TestGetWindowCaptions:
    def test_extracts_captions(self, sample_meta: str) -> None:
        captions = _get_window_captions(sample_meta)
        assert len(captions) == 2
        assert captions[0] == "A dog runs across a field."
        assert captions[1] == "The dog jumps over a fence."

    def test_no_caption_keys(self, tmp_dir: str) -> None:
        meta = {"windows": [{"score": 0.9}]}
        path = os.path.join(tmp_dir, "nocap.json")
        with open(path, "w") as f:
            json.dump(meta, f)
        assert _get_window_captions(path) == []

    def test_empty_windows(self, tmp_dir: str) -> None:
        meta: dict = {"windows": []}
        path = os.path.join(tmp_dir, "empty.json")
        with open(path, "w") as f:
            json.dump(meta, f)
        assert _get_window_captions(path) == []

    def test_whitespace_only_caption_skipped(self, tmp_dir: str) -> None:
        meta = {"windows": [{"caption": "   "}]}
        path = os.path.join(tmp_dir, "ws.json")
        with open(path, "w") as f:
            json.dump(meta, f)
        assert _get_window_captions(path) == []


# ---- _get_source_video ----


class TestGetSourceVideo:
    def test_returns_source_video(self, sample_meta: str) -> None:
        assert _get_source_video(sample_meta) == "/data/video.mp4"

    def test_fallback_to_video_path(self, tmp_dir: str) -> None:
        meta = {"video_path": "/fallback/path.mp4"}
        path = os.path.join(tmp_dir, "fb.json")
        with open(path, "w") as f:
            json.dump(meta, f)
        assert _get_source_video(path) == "/fallback/path.mp4"

    def test_no_keys_returns_unknown(self, tmp_dir: str) -> None:
        path = os.path.join(tmp_dir, "none.json")
        with open(path, "w") as f:
            json.dump({}, f)
        assert _get_source_video(path) == "unknown"


# ---- _cosine_sim ----


class TestCosineSim:
    def test_identical_vectors(self) -> None:
        a = torch.tensor([1.0, 2.0, 3.0])
        assert abs(_cosine_sim(a, a) - 1.0) < 1e-5

    def test_orthogonal_vectors(self) -> None:
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert abs(_cosine_sim(a, b)) < 1e-5

    def test_opposite_vectors(self) -> None:
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([-1.0, 0.0])
        assert abs(_cosine_sim(a, b) + 1.0) < 1e-5

    def test_scaled_vectors_same_direction(self) -> None:
        a = torch.tensor([1.0, 1.0])
        b = torch.tensor([100.0, 100.0])
        assert abs(_cosine_sim(a, b) - 1.0) < 1e-5


# ---- _collect_tasks ----


class TestCollectTasks:
    def test_collects_all_uid_model_pairs(self, caption_dirs: dict[str, str]) -> None:
        tasks = _collect_tasks(["uid-aaa", "uid-bbb"], caption_dirs)
        assert len(tasks) == 4  # 2 uids x 2 models
        labels = {t[1] for t in tasks}
        assert labels == {"qwen25", "nemotron"}
        uids = {t[0] for t in tasks}
        assert uids == {"uid-aaa", "uid-bbb"}

    def test_caption_text_joined(self, caption_dirs: dict[str, str]) -> None:
        tasks = _collect_tasks(["uid-aaa"], caption_dirs)
        for _uid, label, text in tasks:
            assert "Caption for" in text
            assert label in text


# ---- _load_uid_list ----


class TestLoadUidList:
    def test_direct_match(
        self,
        uid_list_file: str,
        embedding_dir: str,
        caption_dirs: dict[str, str],
    ) -> None:
        uids = _load_uid_list(uid_list_file, embedding_dir, caption_dirs)
        assert uids == {"uid-aaa", "uid-bbb"}

    def test_fallback_span_resolution(
        self,
        uid_list_file_with_spans: str,
        embedding_dir: str,
        caption_dirs: dict[str, str],
    ) -> None:
        uids = _load_uid_list(uid_list_file_with_spans, embedding_dir, caption_dirs)
        assert len(uids) == 2

    def test_empty_lines_ignored(
        self,
        tmp_dir: str,
        embedding_dir: str,
        caption_dirs: dict[str, str],
    ) -> None:
        path = os.path.join(tmp_dir, "uids_empty.txt")
        with open(path, "w") as f:
            f.write("uid-aaa\n\n\nuid-bbb\n")
        uids = _load_uid_list(path, embedding_dir, caption_dirs)
        assert uids == {"uid-aaa", "uid-bbb"}


# ---- _load_video_embeddings ----


class TestLoadVideoEmbeddings:
    def test_loads_correct_count(self, embedding_dir: str) -> None:
        cache = _load_video_embeddings(["uid-aaa", "uid-bbb"], embedding_dir)
        assert len(cache) == 2

    def test_correct_shape(self, embedding_dir: str) -> None:
        cache = _load_video_embeddings(["uid-aaa"], embedding_dir)
        assert cache["uid-aaa"].shape == (256,)

    def test_returns_tensors(self, embedding_dir: str) -> None:
        cache = _load_video_embeddings(["uid-ccc"], embedding_dir)
        assert isinstance(cache["uid-ccc"], torch.Tensor)
