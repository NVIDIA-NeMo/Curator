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

"""Architecture tests for shared adapter-backed audio inference behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.audio.inference.base import AdapterInferenceStage
from nemo_curator.stages.audio.inference.sed.stage import SEDInferenceStage
from tests.stages.audio.inference import review_helpers as rh

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def test_asr_and_sed_inherit_one_adapter_stage_base() -> None:
    assert issubclass(ASRStage, AdapterInferenceStage)
    assert issubclass(SEDInferenceStage, AdapterInferenceStage)


def test_common_adapter_infrastructure_is_not_reimplemented() -> None:
    common_methods = {
        "_adapter_class",
        "_adapter_gpu_count",
        "inputs",
        "setup_on_node",
        "setup",
        "teardown",
    }
    assert common_methods.isdisjoint(ASRStage.__dict__)
    assert common_methods.isdisjoint(SEDInferenceStage.__dict__)


def test_worker_sizing_uses_the_processing_stage_override() -> None:
    sed = SEDInferenceStage(adapter_target="package.Adapter", checkpoint_path="/checkpoint.pth")
    asr = ASRStage(adapter_target="package.Adapter", model_id="model")

    assert "num_workers_override" not in SEDInferenceStage.__dataclass_fields__
    assert sed.num_workers() is None
    assert asr.num_workers() is None
    assert sed.with_(num_workers=3).num_workers() == 3
    assert asr.with_(num_workers=3).num_workers() == 3


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_process_batch_accepts_file_waveform_and_auto_residency(
    kind: str, tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    file_waveform = rh.np.arange(12, dtype=rh.np.float32)[None, :]
    resident_waveform = rh.np.arange(20, dtype=rh.np.float32)[None, :] + 100
    audio_path = tmp_path / "source.wav"
    rh._write_audio(audio_path, file_waveform)
    file_stage, file_seen = rh._make_stage(kind, monkeypatch, input_residency="file")
    file_result = file_stage.process_batch([rh.AudioTask(data={"audio_filepath": str(audio_path)})])
    assert len(file_result) == 1
    assert file_seen == [len(file_waveform[0])]
    waveform_stage, waveform_seen = rh._make_stage(kind, monkeypatch, input_residency="waveform")
    waveform_result = waveform_stage.process_batch(
        [rh.AudioTask(data={"waveform": resident_waveform, "sample_rate": rh._SAMPLE_RATE})]
    )
    assert len(waveform_result) == 1
    assert waveform_seen == [len(resident_waveform[0])]
    auto_stage, auto_seen = rh._make_stage(kind, monkeypatch, input_residency="auto")
    auto_result = auto_stage.process_batch(
        [
            rh.AudioTask(
                data={"audio_filepath": str(audio_path), "waveform": resident_waveform, "sample_rate": rh._SAMPLE_RATE}
            )
        ]
    )
    assert len(auto_result) == 1
    assert auto_seen == [len(resident_waveform[0])], "auto must prefer the complete resident pair"
    auto_file_stage, auto_file_seen = rh._make_stage(kind, monkeypatch, input_residency="auto")
    auto_file_result = auto_file_stage.process_batch([rh.AudioTask(data={"audio_filepath": str(audio_path)})])
    assert len(auto_file_result) == 1
    assert auto_file_seen == [len(file_waveform[0])]


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
@rh.pytest.mark.parametrize("residency", ["waveform", "auto"])
@rh.pytest.mark.parametrize(
    "partial", [{"waveform": rh.np.ones((1, 10), dtype=rh.np.float32)}, {"sample_rate": rh._SAMPLE_RATE}]
)
def test_process_batch_rejects_partial_resident_pairs(
    kind: str, residency: str, partial: dict[str, Any], monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    stage, _seen = rh._make_stage(kind, monkeypatch, input_residency=residency)
    partial = {"audio_filepath": "/data/source.wav", **partial}
    with rh.pytest.raises(ValueError, match="incomplete resident audio.*must be provided together"):
        stage.process_batch([rh.AudioTask(data=partial)])


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
@rh.pytest.mark.parametrize(
    "partial", [{"waveform": rh.np.ones((1, 10), dtype=rh.np.float32)}, {"sample_rate": rh._SAMPLE_RATE}]
)
def test_file_residency_ignores_partial_resident_fragments(
    kind: str, partial: dict[str, Any], tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    waveform = rh.np.arange(12, dtype=rh.np.float32)[None, :]
    audio_path = tmp_path / f"{kind}.wav"
    rh._write_audio(audio_path, waveform)
    stage, seen = rh._make_stage(kind, monkeypatch, input_residency="file")
    result = stage.process_batch([rh.AudioTask(data={"audio_filepath": str(audio_path), **partial})])
    assert len(result) == 1
    assert seen == [waveform.shape[-1]]


@rh.pytest.mark.parametrize("cls", [rh.PyAnnoteDiarizationStage, rh.WhisperXVADStage, rh.InferenceSortformerStage])
def test_inference_stages_validate_input_residency_at_construction(cls: type) -> None:
    with rh.pytest.raises(ValueError, match="input_residency must be one of"):
        cls(input_residency="wavefrom")


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_fanout_contract_is_waveform_only_and_blocks_file_consumers(
    kind: str, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    stage, _seen = rh._make_stage(kind, monkeypatch, fanout=True)
    contract = rh.build_contract(stage)
    assert {"waveform", "sample_rate"}.issubset(contract.writes.data_keys)
    assert contract.writes.produces == ["tensor"]
    assert {"audio_filepath", "resampled_audio_filepath"}.issubset(contract.removes_keys)
    removed_containers = {
        "pyannote": {"segments", "overlap_segments"},
        "whisperx": {"vad_segments"},
        "sortformer": {"diar_segments"},
    }
    assert removed_containers[kind].issubset(contract.removes_keys)
    assert contract.cardinality == "1:N fan-out"
    assert contract.iteration_key is None if kind == "pyannote" else contract.iteration_key is not None
    after_fanout = rh.validate_pipeline(
        [stage],
        initial_roles={"audio_filepath"},
        initial_keys={"audio_filepath", "resampled_audio_filepath", *removed_containers[kind]},
        initial_task_type="AudioTask",
    )
    assert removed_containers[kind].isdisjoint(after_fanout.produced_keys)
    file_asr = rh.ASRStage(adapter_target=rh._ASR_TARGET, model_id="mock/model")
    rejected = rh.validate_pipeline(
        [stage, file_asr],
        initial_roles={"audio_filepath"},
        initial_keys={"audio_filepath", "resampled_audio_filepath"},
        initial_task_type="AudioTask",
    )
    assert not rejected.ok
    assert any(issue.stage_index == 1 and issue.code == "key_removed_upstream" for issue in rejected.issues)
    waveform_asr = rh.ASRStage(
        adapter_target=rh._ASR_TARGET, model_id="mock/model", waveform_key="waveform", sample_rate_key="sample_rate"
    )
    accepted = rh.validate_pipeline(
        [stage, waveform_asr],
        initial_roles={"audio_filepath"},
        initial_keys={"audio_filepath", "resampled_audio_filepath"},
        initial_task_type="AudioTask",
    )
    assert accepted.ok
    setattr(stage, "filepath_key" if kind == "sortformer" else "audio_filepath_key", "recording_path")
    custom_contract = rh.build_contract(stage)
    assert {"recording_path", "audio_filepath", "resampled_audio_filepath"}.issubset(custom_contract.removes_keys)


@rh.pytest.mark.parametrize(
    ("cls", "path_key", "container_key"),
    [
        (rh.PyAnnoteDiarizationStage, "audio_filepath_key", "segments_key"),
        (rh.WhisperXVADStage, "audio_filepath_key", "segments_key"),
        (rh.InferenceSortformerStage, "filepath_key", "diar_segments_key"),
    ],
)
def test_fanout_rejects_output_and_path_key_collisions(cls: type, path_key: str, container_key: str) -> None:
    with rh.pytest.raises(ValueError, match="collide with removed full-recording path keys"):
        cls(fanout=True, waveform_key="audio_filepath")
    with rh.pytest.raises(ValueError, match="fan-out output keys must be distinct"):
        cls(fanout=True, waveform_key="samples", sample_rate_key="samples")
    with rh.pytest.raises(ValueError, match="fan-out output keys must be non-empty"):
        cls(fanout=True, original_file_key="")
    with rh.pytest.raises(ValueError, match="collide with removed full-recording path keys"):
        cls(fanout=True, original_file_key="recording_path", **{path_key: "recording_path"})
    with rh.pytest.raises(ValueError, match="collide with removed parent container keys"):
        cls(fanout=True, waveform_key="parent_segments", **{container_key: "parent_segments"})
    stage = cls(
        fanout=True, waveform_key="segment_samples", sample_rate_key="segment_rate", **{path_key: "recording_path"}
    )
    contract = rh.build_contract(stage)
    assert "recording_path" in contract.removes_keys
    assert "recording_path" not in contract.writes.data_keys


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_assert_agent_ready_for_fanout_inference_stages(
    kind: str, tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    waveform = rh.np.arange(12, dtype=rh.np.float32)[None, :]
    audio_path = tmp_path / f"{kind}.wav"
    rh._write_audio(audio_path, waveform)
    stage, _seen = rh._make_stage(kind, monkeypatch, fanout=True)
    rh.assert_agent_ready(
        stage,
        lambda: rh.AudioTask(data={"audio_filepath": str(audio_path)}),
        expected_cardinality="1:N fan-out",
        available_keys={"audio_filepath"},
    )


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
@rh.pytest.mark.parametrize("resident_type", ["numpy", "torch"])
def test_pcm16_resident_fanout_matches_file_amplitude(
    kind: str, resident_type: str, tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    pcm = rh.np.array(
        [[-32768, -24576, -16384, -8192, 0, 8192, 16384, 24576, 32767, -32768, 0, 32767]], dtype=rh.np.int16
    )
    normalized = pcm.astype(rh.np.float32) / rh.np.float32(32768)
    audio_path = tmp_path / f"{kind}-{resident_type}.wav"
    rh.sf.write(audio_path, normalized.T, rh._SAMPLE_RATE, subtype="PCM_16")
    resident = rh.torch.from_numpy(pcm.copy()) if resident_type == "torch" else pcm
    resident_stage, _seen = rh._make_stage(kind, monkeypatch, input_residency="waveform", fanout=True)
    file_stage, _seen = rh._make_stage(kind, monkeypatch, input_residency="file", fanout=True)
    resident_children = resident_stage.process_batch(
        [rh.AudioTask(data={"waveform": resident, "sample_rate": rh._SAMPLE_RATE})]
    )
    file_children = file_stage.process_batch([rh.AudioTask(data={"audio_filepath": str(audio_path)})])
    assert len(resident_children) == len(file_children) == 2
    for resident_child, file_child in zip(resident_children, file_children, strict=True):
        assert resident_child.data["waveform"].dtype == rh.np.float32
        rh.np.testing.assert_array_equal(resident_child.data["waveform"], file_child.data["waveform"])
    asr = rh.ASRStage(
        adapter_target=rh._ASR_TARGET,
        model_id="mock/model",
        waveform_key="waveform",
        sample_rate_key="sample_rate",
        target_sample_rate=rh._SAMPLE_RATE,
        keep_waveform=True,
    )
    asr._adapter = rh.MagicMock()
    asr._adapter.transcribe_batch.return_value = [rh.ASRResult(text="one"), rh.ASRResult(text="two")]
    asr.process_batch(resident_children)
    asr_items = asr._adapter.transcribe_batch.call_args.args[0]
    rh.np.testing.assert_array_equal(asr_items[0]["waveform"], normalized[0, 2:6])
    rh.np.testing.assert_array_equal(asr_items[1]["waveform"], normalized[0, 7:10])


def test_pcm32_and_floating_waveforms_are_canonical_float32() -> None:
    pcm32 = rh.np.array([[-2147483648, -1073741824, 0, 1073741824, 2147483647]], dtype=rh.np.int32)
    expected = pcm32.astype(rh.np.float32) / rh.np.float32(2147483648)
    normalized = rh._channel_first_waveform(pcm32)
    rh.np.testing.assert_array_equal(normalized, expected)
    assert normalized.dtype == rh.np.float32
    floating = rh._channel_first_waveform(rh.np.array([[0.25, -0.5]], dtype=rh.np.float64))
    rh.np.testing.assert_array_equal(floating, rh.np.array([[0.25, -0.5]], dtype=rh.np.float32))
    assert floating.dtype == rh.np.float32


@rh.pytest.mark.parametrize(
    "waveform",
    [
        rh.np.array([[0, 1]], dtype=rh.np.uint16),
        rh.np.array([[0, 1]], dtype=rh.np.int8),
        rh.np.array([[0, 1]], dtype=rh.np.int64),
        rh.np.array([[0, 1]], dtype=rh.np.complex64),
        rh.np.array([["0", "1"]]),
    ],
)
def test_unsupported_resident_waveform_dtypes_are_rejected(waveform: rh.np.ndarray) -> None:
    with rh.pytest.raises(ValueError, match="unsupported resident waveform"):
        rh._channel_first_waveform(waveform)


def test_waveform_identity_hash_streams_a_memoryview(monkeypatch: rh.pytest.MonkeyPatch) -> None:
    payloads: list[bytes | memoryview] = []
    real_sha256 = rh.inference_base.hashlib.sha256

    class RecordingDigest:
        def __init__(self) -> None:
            self.delegate = real_sha256()

        def update(self, payload: bytes | memoryview) -> None:
            payloads.append(payload)
            self.delegate.update(payload)

        def hexdigest(self) -> str:
            return self.delegate.hexdigest()

    monkeypatch.setattr(rh.inference_base.hashlib, "sha256", lambda: RecordingDigest())
    waveform = rh.np.arange(12, dtype=rh.np.float32).reshape(1, -1)
    identity = rh.inference_base._stable_audio_identity({}, waveform, rh._SAMPLE_RATE, source_path=None)
    assert identity.startswith("audio_")
    assert isinstance(payloads[0], memoryview)
    assert payloads[0].nbytes == waveform.nbytes
    assert isinstance(payloads[1], bytes)


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_unsigned_resident_waveforms_are_rejected_by_fanout_stages(
    kind: str, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    stage, _seen = rh._make_stage(kind, monkeypatch, input_residency="waveform", fanout=True)
    with rh.pytest.raises(ValueError, match="unsupported resident waveform integer dtype"):
        stage.process_batch(
            [
                rh.AudioTask(
                    data={"waveform": rh.np.array([[0, 32768]], dtype=rh.np.uint16), "sample_rate": rh._SAMPLE_RATE}
                )
            ]
        )


def test_legacy_positional_signatures_are_exact() -> None:
    expected = {
        rh.PyAnnoteDiarizationStage: [
            "hf_token",
            "model_name",
            "segmentation_batch_size",
            "embedding_batch_size",
            "min_length",
            "max_length",
            "audio_filepath_key",
            "segments_key",
            "overlap_segments_key",
            "name",
            "resources",
            "xenna_num_workers",
            "_pipeline",
            "_vad_model",
            "_rng",
        ],
        rh.InferenceSortformerStage: [
            "model_name",
            "model_path",
            "cache_dir",
            "diar_model",
            "filepath_key",
            "diar_segments_key",
            "rttm_out_dir",
            "chunk_len",
            "chunk_left_context",
            "chunk_right_context",
            "fifo_len",
            "spkcache_update_period",
            "spkcache_len",
            "inference_batch_size",
            "name",
            "batch_size",
            "resources",
        ],
        rh.WhisperXVADStage: [
            "min_length",
            "max_length",
            "vad_onset",
            "vad_offset",
            "segments_key",
            "audio_filepath_key",
            "name",
            "resources",
            "_vad_model",
        ],
    }
    for cls, expected_names in expected.items():
        signature = rh.inspect.signature(cls.__init__)
        parameters = list(signature.parameters.values())[1:]
        positional = [
            parameter.name for parameter in parameters if parameter.kind is rh.inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert positional == expected_names
        keyword_only = {
            parameter.name for parameter in parameters if parameter.kind is rh.inspect.Parameter.KEYWORD_ONLY
        }
        assert {"waveform_key", "sample_rate_key", "input_residency", "fanout"}.issubset(keyword_only)
        bound = signature.bind(None, *range(len(expected_names)))
        assert list(bound.arguments)[1:] == expected_names


def test_static_and_configured_inference_hints_are_truthful(tmp_path: Path) -> None:
    pyannote_static = rh.static_contract(rh.PyAnnoteDiarizationStage)
    assert pyannote_static.gates.requires_internet_first_run
    assert pyannote_static.gates.runtime_secrets == ["HF_TOKEN"]
    assert pyannote_static.gates.writes_to_disk
    assert pyannote_static.gates.output_path_params == []
    assert pyannote_static.gates.per_row_independent is False
    assert {"1:1", "1:N fan-out"}.issubset(pyannote_static.cardinality_options)
    local_pipeline = tmp_path / "pyannote-pipeline"
    local_pipeline.mkdir()
    local_pyannote = rh.build_contract(rh.PyAnnoteDiarizationStage(model_name=str(local_pipeline), write_rttm=False))
    assert local_pyannote.gates.requires_internet_first_run
    assert local_pyannote.gates.runtime_secrets == []
    assert not local_pyannote.gates.writes_to_disk
    whisperx = rh.build_contract(rh.WhisperXVADStage(resources=rh.Resources(gpus=0)))
    assert whisperx.gates.requires_internet_first_run
    assert rh.static_contract(rh.WhisperXVADStage).gates.requires_internet_first_run
    asr_static = rh.static_contract(rh.ASRStage)
    asr_configured = rh.build_contract(
        rh.ASRStage(adapter_target=rh._ASR_TARGET, model_id="mock/model", resources=rh.Resources(gpus=0))
    )
    assert asr_static.gates.requires_gpu
    assert asr_static.gates.requires_internet_first_run
    assert asr_static.gates.per_row_independent is True
    assert asr_static.dispatch == "process_batch"
    assert not asr_configured.gates.requires_gpu
    assert asr_configured.gates.requires_internet_first_run
    assert asr_configured.gates.per_row_independent is True
    default_sortformer = rh.build_contract(rh.InferenceSortformerStage(resources=rh.Resources(gpus=0)))
    local_sortformer = rh.build_contract(
        rh.InferenceSortformerStage(model_path="/models/local.nemo", resources=rh.Resources(gpus=0))
    )
    injected_sortformer = rh.build_contract(
        rh.InferenceSortformerStage(diar_model=rh.MagicMock(), resources=rh.Resources(gpus=0))
    )
    assert default_sortformer.gates.requires_internet_first_run
    assert not local_sortformer.gates.requires_internet_first_run
    assert not injected_sortformer.gates.requires_internet_first_run
    assert rh.static_contract(rh.InferenceSortformerStage).gates.requires_internet_first_run
    assert rh.static_contract(rh.InferenceSortformerStage).gates.output_path_params == ["rttm_out_dir"]
    assert rh.static_contract(rh.InferenceSortformerStage).gates.per_row_independent is False
    with rh.patch("nemo_curator.stages.audio.inference.speaker_diarization.sortformer.snapshot_download") as download:
        rh.InferenceSortformerStage(diar_model=rh.MagicMock()).setup_on_node()
    download.assert_not_called()


def test_waveform_only_identities_are_stable_and_explicit_ids_win(
    tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    waveform = rh.np.arange(12, dtype=rh.np.float32)[None, :]
    changed = waveform + 1
    pyannote, _seen = rh._make_stage("pyannote", monkeypatch, input_residency="waveform", fanout=True)

    def pyannote_identity(audio: rh.np.ndarray, **extra: object) -> tuple[str, str]:
        child = pyannote.process(rh.AudioTask(data={"waveform": audio, "sample_rate": rh._SAMPLE_RATE, **extra}))[0]
        return (child.data["original_file"], child.data["speaker"])

    first_pyannote = pyannote_identity(waveform)
    assert pyannote_identity(waveform) == first_pyannote
    assert pyannote_identity(changed) != first_pyannote
    explicit_original, explicit_speaker = pyannote_identity(waveform, audio_item_id="explicit-id")
    assert explicit_original == "explicit-id"
    assert explicit_speaker.startswith("explicit-id_")
    whisperx, _seen = rh._make_stage("whisperx", monkeypatch, input_residency="waveform", fanout=True)

    def whisperx_identity(audio: rh.np.ndarray, **extra: object) -> str:
        child = whisperx.process(rh.AudioTask(data={"waveform": audio, "sample_rate": rh._SAMPLE_RATE, **extra}))[0]
        return child.data["original_file"]

    first_whisperx = whisperx_identity(waveform)
    assert whisperx_identity(waveform) == first_whisperx
    assert whisperx_identity(changed) != first_whisperx
    assert whisperx_identity(waveform, audio_item_id="explicit-id") == "explicit-id"
    sortformer, _seen = rh._make_stage(
        "sortformer", monkeypatch, input_residency="waveform", rttm_out_dir=str(tmp_path)
    )
    task_data = {"waveform": waveform, "sample_rate": rh._SAMPLE_RATE}
    sortformer.process(rh.AudioTask(data=dict(task_data)))
    first_names = {path.name for path in tmp_path.glob("*.rttm")}
    sortformer.process(rh.AudioTask(data=dict(task_data)))
    assert {path.name for path in tmp_path.glob("*.rttm")} == first_names
    sortformer.process(rh.AudioTask(data={"waveform": changed, "sample_rate": rh._SAMPLE_RATE}))
    assert len(list(tmp_path.glob("*.rttm"))) == 2
    sortformer.process(
        rh.AudioTask(data={"waveform": waveform, "sample_rate": rh._SAMPLE_RATE, "session_name": "explicit-session"})
    )
    assert (tmp_path / "explicit-session.rttm").exists()


def test_file_identity_precedence_preserves_existing_provenance(
    tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    waveform = rh.np.arange(12, dtype=rh.np.float32)[None, :]
    audio_path = tmp_path / "real-source.wav"
    rh._write_audio(audio_path, waveform)
    pyannote, _seen = rh._make_stage("pyannote", monkeypatch, fanout=True)
    pyannote_child = pyannote.process(
        rh.AudioTask(data={"audio_filepath": str(audio_path), "original_file": "/stale/provenance.wav"})
    )[0]
    assert pyannote_child.data["speaker"].startswith("real-source_")
    assert pyannote_child.data["original_file"] == "/stale/provenance.wav"
    pyannote_with_id = pyannote.process(
        rh.AudioTask(
            data={
                "audio_filepath": str(audio_path),
                "audio_item_id": "legacy-item",
                "original_file": "/stale/provenance.wav",
            }
        )
    )[0]
    assert pyannote_with_id.data["speaker"].startswith("legacy-item_")
    assert pyannote_with_id.data["original_file"] == "/stale/provenance.wav"
    pyannote_with_speaker = pyannote.process(
        rh.AudioTask(
            data={
                "audio_filepath": str(audio_path),
                "speaker_id": "legacy-speaker",
                "original_file": "/stale/provenance.wav",
            }
        )
    )[0]
    assert pyannote_with_speaker.data["speaker"].startswith("legacy-speaker_")
    sortformer, _seen = rh._make_stage("sortformer", monkeypatch, fanout=True, rttm_out_dir=str(tmp_path / "rttm"))
    sortformer_child = sortformer.process(
        rh.AudioTask(
            data={
                "audio_filepath": str(audio_path),
                "audio_item_id": "new-item-id",
                "original_file": "/stale/provenance.wav",
            }
        )
    )[0]
    assert (tmp_path / "rttm" / "real-source.rttm").exists()
    assert sortformer_child.data["original_file"] == "/stale/provenance.wav"
    sortformer.process(
        rh.AudioTask(
            data={
                "audio_filepath": str(audio_path),
                "session_name": "explicit-session",
                "audio_item_id": "ignored-item-id",
            }
        )
    )
    assert (tmp_path / "rttm" / "explicit-session.rttm").exists()
    whisperx, _seen = rh._make_stage("whisperx", monkeypatch, fanout=True)
    whisperx_child = whisperx.process(
        rh.AudioTask(data={"audio_filepath": str(audio_path), "original_file": "/stale/provenance.wav"})
    )[0]
    assert whisperx_child.data["original_file"] == "/stale/provenance.wav"


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx"])
def test_configured_file_key_is_strict_unless_fallback_is_enabled(
    kind: str, tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "raw.wav"
    rh._write_audio(audio_path, rh.np.ones((1, 12), dtype=rh.np.float32))
    stage, _seen = rh._make_stage(kind, monkeypatch)
    stage.audio_filepath_key = "clean_audio"
    task = rh.AudioTask(data={"audio_filepath": str(audio_path)})
    with rh.pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([task])
    stage.allow_audio_filepath_fallback = True
    result = stage.process_batch([task])
    assert len(result) == 1


@rh.pytest.mark.parametrize(
    ("stage", "segments_key"),
    [
        (rh.PyAnnoteDiarizationStage(resources=rh.Resources(gpus=0), write_rttm=False), "segments"),
        (rh.InferenceSortformerStage(diar_model=rh.MagicMock(), resources=rh.Resources(gpus=0)), "diar_segments"),
    ],
)
def test_num_speakers_is_disabled_by_default_and_preserves_legacy_aliases(stage: object, segments_key: str) -> None:
    assert stage.num_speakers_key is None
    assert "num_speakers" not in rh.build_contract(stage).writes.data_keys
    assert "num_speakers" not in stage.outputs()[1]
    stage_with_alias = type(stage)(**{segments_key.replace("segments", "segments_key"): "num_speakers"})
    assert stage_with_alias.num_speakers_key is None


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_waveform_materialization_preserves_model_observed_samples(
    kind: str, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    waveform = rh.np.array([[1e-05, -1e-05, 1.25, -1.25]], dtype=rh.np.float32)
    stage, _seen = rh._make_stage(kind, monkeypatch, input_residency="waveform")
    observed: list[rh.np.ndarray] = []
    if kind == "pyannote":
        stage._pipeline = lambda payload, hook=None: (
            hook,
            observed.append(payload["waveform"].numpy().copy()) or rh._FakeAnnotation([]),
        )[1]
    elif kind == "whisperx":
        stage._vad_model.get_vad_segments.side_effect = lambda audio, _max_length, *, sample_rate: (
            sample_rate,
            observed.append(audio.copy()) or [],
        )[1]
    else:
        stage.diar_model.diarize.side_effect = lambda *, audio, batch_size: (
            batch_size,
            observed.append(rh.sf.read(audio[0], dtype="float32", always_2d=True)[0].T.copy()) or [[]],
        )[1]
    stage.process(rh.AudioTask(data={"waveform": waveform, "sample_rate": rh._SAMPLE_RATE}))
    rh.np.testing.assert_array_equal(observed[0], waveform)
    assert rh.build_contract(stage).gates.writes_to_disk


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_fanout_is_non_resumable_until_zero_child_accounting_is_supported(
    kind: str, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    regular, _seen = rh._make_stage(kind, monkeypatch, fanout=False)
    fanout, _seen = rh._make_stage(kind, monkeypatch, fanout=True)
    assert regular.is_resumable is True
    assert fanout.is_resumable is False


@rh.pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_fanout_metadata_uses_normalized_sample_boundaries(kind: str, monkeypatch: rh.pytest.MonkeyPatch) -> None:
    stage, _seen = rh._make_stage(kind, monkeypatch, fanout=True)
    waveform = rh.np.arange(10, dtype=rh.np.float32)[None, :]
    item = {"num_speakers": 1} if kind == "pyannote" else {}
    negative = stage._segment_child_data(item, {"start": -0.2, "end": 0.4}, 0, waveform, 10, "source")
    overrun = stage._segment_child_data(item, {"start": 0.8, "end": 1.5}, 1, waveform, 10, "source")
    assert (negative[stage.start_key], negative[stage.end_key], negative[stage.duration_key]) == (0.0, 0.4, 0.4)
    assert negative[stage.waveform_key].shape[-1] == 4
    assert (overrun[stage.start_key], overrun[stage.end_key]) == (0.8, 1.0)
    assert overrun[stage.duration_key] == rh.pytest.approx(0.2)
    assert overrun[stage.waveform_key].shape[-1] == 2
    with rh.pytest.raises(ValueError, match="end must be >= start"):
        stage._segment_child_data(item, {"start": 0.8, "end": 0.2}, 2, waveform, 10, "source")
