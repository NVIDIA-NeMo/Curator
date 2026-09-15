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

"""Unit tests for incremental continuation planning."""

from types import SimpleNamespace

from nemo_curator import audio_agent as aa
from nemo_curator.audio_agent import _safety, continuation
from nemo_curator.audio_agent.recipe import Recipe

_READER = {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}}  # noqa: S108
_WRITER = {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/out.jsonl"}}  # noqa: S108
_MONO_WF = {"ref": "MonoConversionStage", "params": {"keep_waveform_in_task": True}}
_DUR_WAVE = {"ref": "GetAudioDurationStage", "params": {"input_residency": "waveform"}}
_DUR_FILE = {"ref": "GetAudioDurationStage", "params": {"input_residency": "file"}}
# One of the two pairs that pass state through ``task._metadata`` rather than the row.
_OVERLAP_FILTER = {"ref": "OverlapFilterStage", "params": {}}
_PRETRAIN_METRICS = {"ref": "PretrainMetricsAggregatorStage", "params": {"output_path": "/tmp/metrics.json"}}  # noqa: S108
_SNIPPET = {
    "ref": "SnippetExtractionStage",
    "params": {
        "output_dir": "/tmp/snippets",  # noqa: S108
        "output_audio_tar_path": "/tmp/snippets.tar",  # noqa: S108
        "dry_run": True,
    },
}


def _parent_prefix(new: Recipe, n: int) -> SimpleNamespace:
    """A parent RunRecord stand-in whose stored (redacted) recipe is the exact n-stage
    prefix of ``new`` -- so continuation sees a clean incremental append."""
    dicts = [_safety.redact(s.to_dict(), redact_transcripts=False) for s in new.stages]
    return SimpleNamespace(
        recipe={"stages": dicts[:n]},
        data_fingerprint=None,
        config_hash="PARENT_HASH",
        run_id="run-parent",
        output_paths=["/tmp/parent_out.jsonl"],  # noqa: S108
    )


class TestPlanContinuation:
    def test_unknown_parent_is_graceful(self) -> None:
        # No such parent run -> a JSON dict (full rerun / not-found), never a crash.
        r = aa.plan_continuation({"stages": [_READER, _WRITER]}, "nonexistent-run-id-xyz")
        assert isinstance(r, dict)
        blob = str(r).lower()
        assert any(k in blob for k in ("full_rerun", "not_found", "not found", "missing", "no parent", "run_stages"))


class TestResumeSafety:
    """M2: incremental reuse must not resume a suffix from a disk manifest that cannot
    carry the in-memory waveform the suffix needs."""

    def test_waveform_suffix_forces_full_rerun(self) -> None:
        # Parent leaves an in-memory waveform; the appended stage needs it, but the parent's
        # persisted (disk) output can't carry a tensor -> honest full_rerun, not a bad resume.
        new = Recipe.from_dict({"stages": [_READER, _MONO_WF, _DUR_WAVE]})
        r = continuation.plan_continuation(new, _parent_prefix(new, 2), data_fingerprint=None)
        assert r["mode"] == "full_rerun"
        assert "waveform" in r.get("reason", "")

    def test_file_reload_suffix_stays_incremental(self) -> None:
        # The appended stage reloads audio from file (audio_filepath survives disk) -> reuse.
        new = Recipe.from_dict({"stages": [_READER, _MONO_WF, _DUR_FILE]})
        r = continuation.plan_continuation(new, _parent_prefix(new, 2), data_fingerprint=None)
        assert r["mode"] == "incremental"
        assert r["run_stages"] == ["GetAudioDurationStage"]

    def test_guard_is_fail_safe(self) -> None:
        # An out-of-range boundary never raises and never blocks reuse.
        new = Recipe.from_dict({"stages": [_READER, _MONO_WF, _DUR_WAVE]})
        assert continuation._resume_breaks_on_disk_boundary(new, 99) is None
        assert continuation._resume_breaks_on_disk_boundary(new, 0) is None

    def test_metadata_the_parent_produced_does_not_cross_the_boundary(self) -> None:
        # A manifest holds task.data, so task._metadata is dropped -- and unlike a missing
        # waveform, nothing raises: the reader gets an empty dict and the run "succeeds" with
        # wrong counts. Both sides are declared (metadata_writes / metadata_reads), so the
        # boundary can be checked rather than assumed.
        new = Recipe.from_dict({"stages": [_READER, _OVERLAP_FILTER, _PRETRAIN_METRICS]})
        r = continuation.plan_continuation(new, _parent_prefix(new, 2), data_fingerprint=None)
        assert r["mode"] == "full_rerun"
        assert "pretrain_long_form" in r["reason"]
        assert "PretrainMetricsAggregatorStage" in r["reason"]

    def test_a_suffix_that_remakes_the_key_is_not_blocked(self) -> None:
        # The tail plans and then aggregates its own counters, so nothing needed crosses the
        # boundary. Refusing here would penalize a suffix for reading what it just wrote.
        new = Recipe.from_dict({"stages": [_READER, _OVERLAP_FILTER, _OVERLAP_FILTER, _PRETRAIN_METRICS]})
        assert continuation._resume_breaks_on_disk_boundary(new.freeze(), 2) is None

    def test_task_id_dependent_durable_suffix_forces_full_rerun(self) -> None:
        new = Recipe.from_dict({"stages": [_READER, _DUR_FILE, _SNIPPET]})

        result = continuation.plan_continuation(
            new,
            _parent_prefix(new, 2),
            data_fingerprint=None,
        )

        assert result["mode"] == "full_rerun"
        assert "stable framework task.task_id" in result["reason"]
        assert "SnippetExtractionStage" in result["reason"]
