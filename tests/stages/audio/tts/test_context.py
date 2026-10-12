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


import numpy as np
import pytest

from nemo_curator.stages.audio.tts.context import (
    ContextSelectionStage,
    assign_contexts,
    compute_text_overlap,
    empty_context,
    has_usable_context,
    rejection_reason,
)
from nemo_curator.tasks import AudioTask


def seg(member: str, text: str, duration: float = 4.0, speaker: str | None = "v_speaker_0", **extra: object) -> dict:
    segment = {
        "duration": duration,
        "GranaryV2": {"pnc_text": text, "tn_raw": text.lower()},
        "GranaryHifi": {
            "AudioMetadata": {
                "tar_key": "t.tar",
                "tar_member": member,
                "tar_member_offset": 512,
                "tar_member_bytes": 9,
            }
        },
    }
    if speaker is not None:
        segment["speaker_cluster"] = {"speaker_id": speaker}
    segment.update(extra)
    return segment


def unit(angle_deg: float) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)


# ------------------------------------------------------------------------------------------------- text overlap


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 1.0),
        ("!!!", "...", 1.0),
        ("hello", "", 0.0),
        ("Hello, World!", "hello world", 1.0),
        ("a b", "c d", 0.0),
        ("one two three", "one two three four five six", 1.0),  # containment, smaller set has 3 words
        ("one two", "one two three four", 0.5),  # no containment (<3 words): max(jaccard 0.5, bigram 1/3)
        ("你好世界", "世界", 0.5),  # CJK chars are tokens: {你,好,世,界} vs {世,界}; bigram 1/3
    ],
)
def test_compute_text_overlap(a: str, b: str, expected: float) -> None:
    assert compute_text_overlap(a, b) == pytest.approx(expected)


def test_overlap_mixed_cjk_token_split() -> None:
    # "abc你def" -> ["abc", "你", "def"]
    assert compute_text_overlap("abc你def", "abc 你 def") == 1.0


# ------------------------------------------------------------------------------------------------ selection


def run_assign(
    segments: list[dict], embeddings: list, errors: list | None = None, row: dict | None = None, **kw: object
) -> list[dict]:
    assign_contexts(row or {}, segments, embeddings, errors or [None] * len(segments), **kw)
    return [s["context"] for s in segments]


def test_picks_most_similar_valid_candidate() -> None:
    segments = [
        seg("t.wav", "Target words here"),
        seg("near.wav", "Other sentence entirely"),
        seg("far.wav", "Yet another one"),
    ]
    contexts = run_assign(segments, [unit(0), unit(10), unit(40)])
    assert contexts[0]["context_recording_id"] == "near.wav"
    assert contexts[0]["context_audio_speaker_similarity"] == pytest.approx(round(np.cos(np.deg2rad(10)), 6))
    assert contexts[0]["context_audio_offset"] == 512
    assert contexts[0]["context_audio_bytes"] == 9
    assert contexts[0]["context_audio_text"] == "Other sentence entirely"
    assert contexts[0]["context_audio_tn_raw"] == "other sentence entirely"
    assert contexts[0]["error"] is None
    assert contexts[1]["context_recording_id"] == "t.wav"
    assert contexts[2]["context_recording_id"] == "near.wav"


def test_skips_short_overlapping_and_missing_metadata_in_similarity_order() -> None:
    segments = [
        seg("t.wav", "the quick brown fox"),
        seg("short.wav", "different words", duration=2.9),
        seg("dup.wav", "The quick brown fox!"),
        seg("nometa.wav", "unrelated text"),
        seg("ok.wav", "something else"),
    ]
    del segments[3]["GranaryHifi"]["AudioMetadata"]["tar_member_bytes"]
    contexts = run_assign(segments, [unit(0), unit(1), unit(2), unit(3), unit(30)])
    assert contexts[0]["context_recording_id"] == "ok.wav"


@pytest.mark.parametrize(
    ("variant", "reason"),
    [
        ("single", "no_context_segment_for_speaker"),
        ("dissimilar", "no_context_audio_below_min_ssim"),
        ("short", "no_context_audio_below_min_duration"),
        ("overlap", "no_context_audio_text_overlap"),
        ("nometa", "no_context_audio_missing_metadata"),
        ("mixed", "no_context_audio_after_filters"),
    ],
)
def test_rejection_reasons(variant: str, reason: str) -> None:
    target = seg("t.wav", "alpha beta gamma")
    if variant == "single":
        segments, embs = [target], [unit(0)]
    elif variant == "dissimilar":
        segments, embs = [target, seg("a.wav", "x y"), seg("b.wav", "z w")], [unit(0), unit(80), unit(90)]
    elif variant == "short":
        segments, embs = [target, seg("a.wav", "x y", 1.0), seg("b.wav", "z w", 2.0)], [unit(0), unit(5), unit(6)]
    elif variant == "overlap":
        segments, embs = [target, seg("a.wav", "alpha beta gamma")], [unit(0), unit(5)]
    elif variant == "nometa":
        other = seg("a.wav", "x y")
        other["GranaryV2"].pop("pnc_text")
        segments, embs = [target, other], [unit(0), unit(5)]
    else:
        segments, embs = [target, seg("a.wav", "x y", 1.0), seg("b.wav", "z w")], [unit(0), unit(5), unit(80)]
    assert run_assign(segments, embs)[0] == empty_context(reason)


def test_groups_by_speaker_and_reports_embedding_errors() -> None:
    segments = [
        seg("a.wav", "one two", speaker="s0"),
        seg("b.wav", "three four", speaker="s1"),
        seg("c.wav", "five six", speaker="s0"),
        seg("d.wav", "seven", speaker=None),
        seg("e.wav", "eight", speaker="s0"),
        seg("f.wav", "nine", speaker="s0"),
    ]
    contexts = run_assign(
        segments,
        [unit(0), unit(0), unit(5), unit(0), unit(0), None],
        errors=[None, None, None, None, "audio_load_failed", None],
    )
    assert contexts[0]["context_recording_id"] == "c.wav"
    assert contexts[1] == empty_context("no_context_segment_for_speaker")
    assert contexts[3] == empty_context("missing_speaker_id")
    assert contexts[4] == empty_context("audio_load_failed")
    assert contexts[5] == empty_context("embedding_failed")


def test_sample_scope_speaker_for_librilight() -> None:
    segments = [seg("a.wav", "one two", speaker=None), seg("b.wav", "three four", speaker=None)]
    contexts = run_assign(
        segments,
        [unit(0), unit(5)],
        row={"speaker_id": "1234"},
        speaker_id_key="speaker_id",
        speaker_id_scope="sample",
    )
    assert [c["context_recording_id"] for c in contexts] == ["b.wav", "a.wav"]
    with pytest.raises(ValueError, match="speaker_id_scope"):
        assign_contexts({}, [], [], [], speaker_id_scope="video")


def test_rejection_reason_priority() -> None:
    base = {"total": 2, "below_min_ssim": 0, "below_min_duration": 0, "text_overlap": 0, "missing_metadata": 0}
    assert rejection_reason({**base, "total": 0}) == "no_context_segment_for_speaker"
    assert rejection_reason({**base, "below_min_ssim": 2}) == "no_context_audio_below_min_ssim"
    assert rejection_reason({**base, "below_min_ssim": 1, "below_min_duration": 1}) == "no_context_audio_after_filters"


# ---------------------------------------------------------------------------------------------------- stage


class FakeEmbedder:
    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors
        self.model = object()
        self.calls = 0

    def download(self) -> None:
        pass

    def load(self) -> None:
        pass

    def embed_segments(self, segments: list[dict], _config: object = None) -> tuple[list, list]:
        self.calls += 1
        members = [s["GranaryHifi"]["AudioMetadata"]["tar_member"] for s in segments]
        return [self.vectors.get(m) for m in members], [
            None if m in self.vectors else "audio_load_failed" for m in members
        ]


def test_stage_with_injected_embedder_and_drop_without_context() -> None:
    embedder = FakeEmbedder({"a.wav": unit(0), "b.wav": unit(10), "c.wav": unit(85)})
    stage = ContextSelectionStage(embedder=embedder, drop_without_context=True)
    stage.setup_on_node()
    stage.setup()
    row = {
        "segments": [seg("a.wav", "one two"), seg("b.wav", "three four"), seg("c.wav", "five six"), seg("x.wav", "q")]
    }
    out = stage.process(AudioTask(data=row)).data
    assert [s["GranaryHifi"]["AudioMetadata"]["tar_member"] for s in out["segments"]] == ["a.wav", "b.wav"]
    assert all(has_usable_context(s) for s in out["segments"])
    assert embedder.calls == 1


def test_stage_uses_precomputed_embeddings_and_removes_them() -> None:
    embedder = FakeEmbedder({})
    stage = ContextSelectionStage(embedder=embedder, embeddings_key="_emb", output_key="GranaryHifi.Ctx")
    segments = [seg("a.wav", "one two", _emb=[1.0, 0.0]), seg("b.wav", "three four", _emb=[0.9, 0.1])]
    out = stage.process_batch([AudioTask(data={"segments": segments})])[0].data
    assert embedder.calls == 0
    assert out["segments"][0]["GranaryHifi"]["Ctx"]["context_recording_id"] == "b.wav"
    assert all("_emb" not in s for s in out["segments"])
    assert stage.process_batch([]) == []


def test_stage_ignores_rows_without_segments_and_validates_scope() -> None:
    stage = ContextSelectionStage(embedder=FakeEmbedder({}))
    assert stage.process(AudioTask(data={"x": 1})).data == {"x": 1}
    with pytest.raises(ValueError, match="speaker_id_scope"):
        ContextSelectionStage(speaker_id_scope="video")


def test_has_usable_context() -> None:
    assert not has_usable_context({})
    assert not has_usable_context({"context": empty_context("x")})
    ok = {"context": {"error": None, "context_recording_id": "a", "context_audio_offset": 0, "context_audio_bytes": 1}}
    assert has_usable_context(ok)
