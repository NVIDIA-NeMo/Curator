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

"""Unit tests for SpeakerEmbeddingLhotseStage helper functions and merge utility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.speaker_id.speaker_embedding_lhotse import (
    _expand_nemo_path,
    _extract_shard_id,
    _tqdm_enabled,
    _worker_tag,
    merge_shard_embeddings,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestExpandNemoPath:
    def test_brace_pattern(self) -> None:
        result = _expand_nemo_path("manifest_{0..2}.json")
        assert result == ["manifest_0.json", "manifest_1.json", "manifest_2.json"]

    def test_nemo_op_cl_pattern(self) -> None:
        result = _expand_nemo_path("manifest__OP_0..2_CL_.json")
        assert result == ["manifest_0.json", "manifest_1.json", "manifest_2.json"]

    def test_no_pattern_returns_as_is(self) -> None:
        result = _expand_nemo_path("manifest_5.json")
        assert result == ["manifest_5.json"]

    def test_single_element_range(self) -> None:
        result = _expand_nemo_path("file_{3..3}.tar")
        assert result == ["file_3.tar"]


class TestExtractShardId:
    def test_manifest_path(self) -> None:
        assert _extract_shard_id("manifest_25.json") == "25"

    def test_audio_tar(self) -> None:
        assert _extract_shard_id("audio_3.tar") == "3"

    def test_full_path(self) -> None:
        assert _extract_shard_id("/data/manifests/manifest_10.json") == "10"

    def test_no_numeric_suffix(self) -> None:
        result = _extract_shard_id("manifest.json")
        assert result == "manifest"


class TestTqdmEnabled:
    def test_env_override_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURATOR_TQDM", "1")
        assert _tqdm_enabled(default=False) is True

    def test_env_override_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURATOR_TQDM", "0")
        assert _tqdm_enabled(default=True) is False

    def test_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CURATOR_TQDM", raising=False)
        assert _tqdm_enabled(default=False) is False


class TestWorkerTag:
    def test_with_cuda_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        assert _worker_tag() == "gpu3"

    def test_without_cuda_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert _worker_tag().startswith("pid")


class TestMergeShardEmbeddings:
    def test_merge_npz(self, tmp_path: Path) -> None:
        for i in range(3):
            ids = np.array([f"cut_{i}_0", f"cut_{i}_1"], dtype=object)
            embs = np.random.default_rng(42).standard_normal((2, 64)).astype(np.float32)
            np.savez(tmp_path / f"embeddings_{i}.npz", cut_ids=ids, embeddings=embs)

        merged = merge_shard_embeddings(str(tmp_path), output_format="npz")
        data = np.load(merged, allow_pickle=True)

        assert len(data["cut_ids"]) == 6
        assert data["embeddings"].shape == (6, 64)

    def test_merge_pt(self, tmp_path: Path) -> None:
        for i in range(2):
            ids = [f"cut_{i}_0"]
            embs = torch.randn(1, 64)
            torch.save({"cut_ids": ids, "embeddings": embs}, tmp_path / f"embeddings_{i}.pt")

        merged = merge_shard_embeddings(str(tmp_path), output_format="pt")
        data = torch.load(merged, weights_only=False)

        assert len(data["cut_ids"]) == 2
        assert data["embeddings"].shape == (2, 64)

    def test_no_files_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No embeddings"):
            merge_shard_embeddings(str(tmp_path))

    def test_custom_output_path(self, tmp_path: Path) -> None:
        np.savez(
            tmp_path / "embeddings_0.npz",
            cut_ids=np.array(["a"], dtype=object),
            embeddings=np.random.default_rng(42).standard_normal((1, 8)).astype(np.float32),
        )
        custom = str(tmp_path / "custom" / "merged.npz")
        result = merge_shard_embeddings(str(tmp_path), merged_path=custom)
        assert result == custom
