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

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nemo_curator.stages.audio.tts.speaker_clustering import (
    CLUSTERING_FAILED,
    SpeakerClusteringStage,
    cluster_embeddings,
)
from nemo_curator.tasks import AudioTask
from tests.stages.audio.tts.test_audio_io import make_tar_segments


class FakeEmbedder:
    """Returns a fixed vector per segment (looked up by ``vec`` key) or an error code."""

    def __init__(self) -> None:
        self.seen: list[list[dict[str, Any]]] = []

    def download(self) -> None:
        pass

    def load(self) -> None:
        pass

    def embed_segments(
        self, segments: list[dict[str, Any]], _config: object = None
    ) -> tuple[list[np.ndarray | None], list[str | None]]:
        self.seen.append(segments)
        vectors = [None if "err" in s else np.asarray(s["vec"], dtype=np.float32) for s in segments]
        return vectors, [s.get("err") for s in segments]


def seg(vec: list[float] | None, num_speakers: Any = 1, **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    segment: dict[str, Any] = {"duration": 3.0, "vec": vec, **extra}
    if num_speakers != "absent":
        segment["diarization"] = {"num_speakers": num_speakers}
    return segment


def test_cluster_embeddings_two_speakers() -> None:
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    emb = np.array([b, a, [0.95, 0.05, 0.0], [0.05, 0.95, 0.0]])
    labels, sims, margins = cluster_embeddings(emb, 0.4)
    assert labels[0] == labels[3]
    assert labels[1] == labels[2]
    assert labels[0] != labels[1]
    assert sorted(set(labels.tolist())) == [0, 1]
    assert all(0.99 < s <= 1.0 + 1e-6 for s in sims)
    assert all(m is not None and m > 0.9 for m in margins)


def test_cluster_embeddings_single_and_one_cluster() -> None:
    labels, sims, margins = cluster_embeddings(np.array([[3.0, 4.0]]))
    assert labels.tolist() == [0]
    assert sims == [pytest.approx(1.0)]
    assert margins == [None]
    labels, _, margins = cluster_embeddings(np.array([[1.0, 0.0], [0.99, 0.01]]), 0.4)
    assert labels.tolist() == [0, 0]
    assert margins == [None, None]
    labels, sims, margins = cluster_embeddings(np.zeros((0, 2)))
    assert labels.size == 0
    assert sims == []
    assert margins == []


def test_stage_eligibility_and_naming() -> None:
    segments = [
        seg([1.0, 0.0]),
        seg([0.0, 1.0]),
        seg([1.0, 0.0], num_speakers=2),
        seg([1.0, 0.0], num_speakers="absent"),
        seg([1.0, 0.0], num_speakers="x"),
        seg([0.98, 0.02]),
        seg(None, err="audio_load_failed"),
        "junk",
    ]
    embedder = FakeEmbedder()
    stage = SpeakerClusteringStage(embedder=embedder)
    stage.setup()
    out = stage.process(AudioTask(data={"orig_sample_id": "vid42", "segments": segments}))
    ann = [s["speaker_cluster"] if isinstance(s, dict) else None for s in out.data["segments"]]
    assert len(embedder.seen[0]) == 4
    assert ann[0]["speaker_id"] == ann[5]["speaker_id"]
    assert {ann[0]["speaker_id"], ann[1]["speaker_id"]} == {"vid42_speaker_0", "vid42_speaker_1"}
    assert ann[0]["error"] is None
    assert ann[0]["cluster_similarity"] == round(ann[0]["cluster_similarity"], 6)
    assert ann[1]["cluster_similarity"] == pytest.approx(1.0)
    assert ann[1]["cluster_margin"] is not None
    assert ann[2] == {
        "speaker_id": None,
        "cluster_similarity": None,
        "cluster_margin": None,
        "error": "not_single_speaker",
    }
    assert ann[3]["error"] == "missing_num_speakers"
    assert ann[4]["error"] == "invalid_num_speakers"
    assert ann[6]["error"] == "audio_load_failed"
    assert ann[7] is None


def test_speaker_id_prefix_fallback_and_flat_rows() -> None:
    stage = SpeakerClusteringStage(embedder=FakeEmbedder(), num_speakers_key="diar.n")
    row = {"sample_id": "s1", "segments": [{"vec": [1.0, 0.0], "diar": {"n": 1}}]}
    stage.process(AudioTask(data=row))
    assert row["segments"][0]["speaker_cluster"]["speaker_id"] == "s1_speaker_0"
    flat = {"vec": [1.0], "diar": {"n": 1}}
    assert stage.process(AudioTask(data=flat)).data == {"vec": [1.0], "diar": {"n": 1}}


def test_clustering_failure_is_annotated(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_curator.stages.audio.tts.speaker_clustering as mod

    def boom(*_args: object, **_kwargs: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(mod, "cluster_embeddings", boom)
    row = {"orig_sample_id": "v", "segments": [seg([1.0, 0.0]), seg([0.0, 1.0])]}
    SpeakerClusteringStage(embedder=FakeEmbedder()).process(AudioTask(data=row))
    assert all(s["speaker_cluster"]["error"] == CLUSTERING_FAILED for s in row["segments"])


def test_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="distance_threshold"):
        SpeakerClusteringStage(distance_threshold=0.0)


@pytest.mark.gpu
def test_real_titanet_smoke(tmp_path: Path) -> None:
    """Real TitaNet on synthetic clips: identical audio must land in one cluster with similarity ~1."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("nemo.collections.asr")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    stage = SpeakerClusteringStage(model_path=os.environ.get("TITANET_NEMO"))
    try:
        stage.setup()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"TitaNet weights unavailable: {exc}")
    segments = make_tar_segments(tmp_path, [("a.wav", 3.0, 16000), ("b.wav", 3.0, 16000)])
    for segment in segments:
        segment["diarization"] = {"num_speakers": 1}
    stage.process(AudioTask(data={"orig_sample_id": "smoke", "segments": segments}))
    ann = [s["speaker_cluster"] for s in segments]
    assert all(a["error"] is None for a in ann)
    assert ann[0]["speaker_id"] == ann[1]["speaker_id"] == "smoke_speaker_0"
    assert ann[0]["cluster_similarity"] == pytest.approx(1.0, abs=1e-3)
