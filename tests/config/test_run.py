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

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nemo_curator.config.run import create_executor_from_yaml, create_pipeline_from_yaml, main
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter


def test_pipeline_with_jsonl_reader_stage():
    cfg = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.text.io.reader.JsonlReader",
                    "file_paths": "./data",
                    "files_per_partition": 1,
                    "blocksize": "256MB",
                    "fields": ["text", "id"],
                }
            ]
        }
    )

    pipeline = create_pipeline_from_yaml(cfg)

    assert isinstance(pipeline, Pipeline)
    assert pipeline.name == "yaml_pipeline"
    assert len(pipeline.stages) == 1
    assert isinstance(pipeline.stages[0], JsonlReader)


def test_pipeline_with_jsonl_reader_stage_string_vs_list():
    cfg_string = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.text.io.reader.JsonlReader",
                    "file_paths": "./data",
                    "files_per_partition": None,
                    "blocksize": None,
                    "fields": None,
                }
            ]
        }
    )

    cfg_list = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.text.io.reader.JsonlReader",
                    "file_paths": ["./data1", "./data2"],
                    "files_per_partition": None,
                    "blocksize": None,
                    "fields": None,
                }
            ]
        }
    )

    pipeline_string = create_pipeline_from_yaml(cfg_string)
    pipeline_list = create_pipeline_from_yaml(cfg_list)

    assert isinstance(pipeline_string, Pipeline)
    assert isinstance(pipeline_list, Pipeline)
    assert isinstance(pipeline_string.stages[0], JsonlReader)
    assert isinstance(pipeline_list.stages[0], JsonlReader)


def test_pipeline_with_parquet_reader_stage():
    cfg = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.text.io.reader.ParquetReader",
                    "file_paths": "./data",
                    "files_per_partition": 2,
                    "blocksize": "512MB",
                    "fields": ["text"],
                }
            ]
        }
    )

    pipeline = create_pipeline_from_yaml(cfg)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.stages) == 1
    assert isinstance(pipeline.stages[0], ParquetReader)


def test_pipeline_with_jsonl_writer_stage():
    cfg = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.text.io.writer.JsonlWriter",
                    "path": "./output",
                    "fields": ["text", "id"],
                }
            ]
        }
    )

    pipeline = create_pipeline_from_yaml(cfg)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.stages) == 1
    assert isinstance(pipeline.stages[0], JsonlWriter)


def test_pipeline_with_parquet_writer_stage():
    cfg = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.text.io.writer.ParquetWriter",
                    "path": "./output",
                    "fields": ["text"],
                }
            ]
        }
    )

    pipeline = create_pipeline_from_yaml(cfg)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.stages) == 1
    assert isinstance(pipeline.stages[0], ParquetWriter)


def test_pipeline_with_hydra_instantiated_stage():
    from nemo_curator.stages.text.filters import ScoreFilter
    from nemo_curator.stages.text.filters.heuristic import NonAlphaNumericFilter

    cfg = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.text.filters.score_filter.ScoreFilter",
                    "filter_obj": {
                        "_target_": "nemo_curator.stages.text.filters.heuristic.string.NonAlphaNumericFilter",
                        "max_non_alpha_numeric_to_text_ratio": 0.25,
                    },
                    "text_field": "text",
                    "score_field": None,
                }
            ]
        }
    )

    pipeline = create_pipeline_from_yaml(cfg)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.stages) == 1
    assert isinstance(pipeline.stages[0], ScoreFilter)
    assert len(pipeline.stages[0].filter_obj) == 1
    assert isinstance(pipeline.stages[0].filter_obj[0], NonAlphaNumericFilter)
    assert pipeline.stages[0].filter_obj[0]._cutoff == 0.25  # self._cutoff = max_non_alpha_numeric_to_text_ratio
    assert pipeline.stages[0].text_field == ["text"]  # ScoreFilter converts text_field to list
    assert pipeline.stages[0].score_field == [None]  # ScoreFilter converts score_field to list


def test_pipeline_with_hydra_instantiated_resources():
    from nemo_curator.stages.audio.metrics.wer import GetPairwiseWerStage

    cfg = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.audio.metrics.wer.GetPairwiseWerStage",
                    "resources": {
                        "_target_": "nemo_curator.stages.resources.Resources",
                        "gpus": 2.0,
                    },
                }
            ]
        }
    )

    pipeline = create_pipeline_from_yaml(cfg, log_config=False)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.stages) == 1
    assert isinstance(pipeline.stages[0], GetPairwiseWerStage)
    assert pipeline.stages[0].resources.gpus == 2.0


def test_pipeline_with_multiple_stages():
    from nemo_curator.stages.text.modifiers import Modify
    from nemo_curator.stages.text.modifiers.string import UrlRemover

    cfg = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.text.io.reader.JsonlReader",
                    "file_paths": "./data",
                    "files_per_partition": 1,
                    "blocksize": None,
                    "fields": None,
                },
                {
                    "_target_": "nemo_curator.stages.text.modifiers.modifier.Modify",
                    "modifier_fn": {"_target_": "nemo_curator.stages.text.modifiers.string.url_remover.UrlRemover"},
                    "input_fields": "text",
                },
                {
                    "_target_": "nemo_curator.stages.text.io.writer.ParquetWriter",
                    "path": "./output",
                    "fields": None,
                },
            ]
        }
    )

    pipeline = create_pipeline_from_yaml(cfg)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.stages) == 3
    assert isinstance(pipeline.stages[0], JsonlReader)
    assert isinstance(pipeline.stages[1], Modify)
    assert len(pipeline.stages[1].modifier_fn) == 1
    assert isinstance(pipeline.stages[1].modifier_fn[0], UrlRemover)
    assert pipeline.stages[1].input_fields == "text"
    assert isinstance(pipeline.stages[2], ParquetWriter)


def test_pipeline_with_empty_stages_list():
    cfg = OmegaConf.create({"stages": []})

    pipeline = create_pipeline_from_yaml(cfg)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.stages) == 0
    assert pipeline.name == "yaml_pipeline"


@patch("hydra.utils.instantiate")
def test_workflow_creation_single_workflow(mock_instantiate: MagicMock):
    mock_workflow = MagicMock()
    mock_instantiate.return_value = mock_workflow

    cfg = OmegaConf.create(
        {
            "workflow": [
                {
                    "_target_": "nemo_curator.stages.deduplication.exact.workflow.ExactDeduplicationWorkflow",
                    "input_path": "./data",
                    "output_path": "./output",
                    "input_filetype": "jsonl",
                    "text_field": "text",
                }
            ]
        }
    )

    result = create_pipeline_from_yaml(cfg)

    assert result == mock_workflow
    mock_instantiate.assert_called_once_with(cfg.workflow[0])


def test_multiple_workflows_raises_error():
    cfg = OmegaConf.create(
        {
            "workflow": [
                {
                    "_target_": "nemo_curator.stages.deduplication.exact.workflow.ExactDeduplicationWorkflow",
                    "input_path": "./data1",
                },
                {
                    "_target_": "nemo_curator.stages.deduplication.fuzzy.workflow.FuzzyDeduplicationWorkflow",
                    "input_path": "./data2",
                },
            ]
        }
    )

    with pytest.raises(RuntimeError, match="One workflow should be defined in the YAML configuration"):
        create_pipeline_from_yaml(cfg)


def test_no_stages_or_workflow_raises_error():
    cfg = OmegaConf.create({"some_other_config": "value"})

    with pytest.raises(RuntimeError, match="Invalid YAML configuration"):
        create_pipeline_from_yaml(cfg)


def test_empty_config_raises_error():
    cfg = OmegaConf.create({})

    with pytest.raises(RuntimeError, match="Invalid YAML configuration"):
        create_pipeline_from_yaml(cfg)


def test_empty_workflow_list_raises_error():
    cfg = OmegaConf.create({"workflow": []})

    with pytest.raises(RuntimeError, match="One workflow should be defined in the YAML configuration"):
        create_pipeline_from_yaml(cfg)


def test_both_stages_and_workflow_raises_error():
    cfg = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.text.io.reader.JsonlReader",
                    "file_paths": "./data",
                }
            ],
            "workflow": [{"_target_": "nemo_curator.stages.deduplication.exact.workflow.ExactDeduplicationWorkflow"}],
        }
    )

    with pytest.raises(RuntimeError, match="Both stages and workflow"):
        create_pipeline_from_yaml(cfg)


def test_executor_uses_pipeline_default_without_executor_config():
    cfg = OmegaConf.create({"stages": []})

    assert create_executor_from_yaml(cfg) is None


@patch("hydra.utils.get_class")
def test_executor_configures_xenna_execution_mode(mock_get_class: MagicMock):
    executor_cls = MagicMock()
    mock_get_class.return_value = executor_cls
    cfg = OmegaConf.create({"backend": "xenna", "execution_mode": "batch"})

    executor = create_executor_from_yaml(cfg)

    mock_get_class.assert_called_once_with("nemo_curator.backends.xenna.XennaExecutor")
    executor_cls.assert_called_once_with(config={"execution_mode": "batch"})
    assert executor is executor_cls.return_value


@patch("hydra.utils.get_class")
def test_execution_mode_defaults_to_xenna_backend(mock_get_class: MagicMock):
    executor_cls = MagicMock()
    mock_get_class.return_value = executor_cls
    cfg = OmegaConf.create({"execution_mode": "streaming"})

    create_executor_from_yaml(cfg)

    mock_get_class.assert_called_once_with("nemo_curator.backends.xenna.XennaExecutor")
    executor_cls.assert_called_once_with(config={"execution_mode": "streaming"})


@patch("hydra.utils.get_class")
def test_ray_data_does_not_receive_xenna_execution_mode(mock_get_class: MagicMock):
    executor_cls = MagicMock()
    mock_get_class.return_value = executor_cls
    cfg = OmegaConf.create({"backend": "ray_data", "execution_mode": "batch"})

    create_executor_from_yaml(cfg)

    mock_get_class.assert_called_once_with("nemo_curator.backends.ray_data.RayDataExecutor")
    executor_cls.assert_called_once_with()


def test_unknown_executor_backend_raises_error():
    cfg = OmegaConf.create({"backend": "unknown"})

    with pytest.raises(ValueError, match="Unknown backend 'unknown'"):
        create_executor_from_yaml(cfg)


def test_unknown_xenna_execution_mode_raises_error():
    cfg = OmegaConf.create({"backend": "xenna", "execution_mode": "unknown"})

    with pytest.raises(ValueError, match="Unknown Xenna execution mode 'unknown'"):
        create_executor_from_yaml(cfg)


@patch("nemo_curator.config.run.create_executor_from_yaml")
@patch("nemo_curator.config.run.create_pipeline_from_yaml")
@patch("nemo_curator.config.run.create_ray_client_from_yaml")
def test_main_passes_configured_executor_to_pipeline(
    mock_create_ray_client: MagicMock,
    mock_create_pipeline: MagicMock,
    mock_create_executor: MagicMock,
):
    cfg = OmegaConf.create({"stages": []})
    executor = mock_create_executor.return_value

    main.__wrapped__(cfg)

    mock_create_ray_client.return_value.start.assert_called_once_with()
    mock_create_pipeline.return_value.run.assert_called_once_with(executor=executor)
    mock_create_ray_client.return_value.stop.assert_called_once_with()


def test_qwen_tutorial_yaml_matches_reference_runner_config():
    config_dir = Path(__file__).parents[2] / "tutorials" / "audio" / "qwen_omni_inprocess"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl",
                "pred_text_key=custom_prediction",
                "model_revision=abc123",
            ],
        )

    pipeline = create_pipeline_from_yaml(cfg, log_config=False)
    resample_stage = pipeline.stages[1]
    stage = pipeline.stages[2]
    executor = create_executor_from_yaml(cfg)

    assert resample_stage.__class__.__name__ == "ResampleAudioStage"
    assert "sample_rate" not in cfg
    assert Path(resample_stage.resampled_audio_dir).parts[-2:] == ("qwen_omni_workspace", "audio_resampled")
    assert resample_stage.target_sample_rate == 16000
    assert resample_stage.target_format == "wav"
    assert resample_stage.target_nchannels == 1
    assert stage.audio_filepath_key == "resampled_audio_filepath"
    assert stage.inputs()[1] == ["resampled_audio_filepath"]
    assert stage.pred_text_key == "custom_prediction"
    assert stage.extras_key is None
    assert stage.outputs() == ([], ["custom_prediction", "_skipme", "additional_notes"])
    assert stage.default_language is None
    assert stage.supported_language_codes == [
        "en",
        "zh",
        "ko",
        "ja",
        "de",
        "ru",
        "it",
        "fr",
        "es",
        "pt",
        "ms",
        "nl",
        "id",
        "tr",
        "vi",
        "yue",
        "ar",
        "ur",
    ]
    assert stage.batch_size == 32
    assert stage.resources.gpus == 2
    assert dict(stage.adapter_kwargs) == {
        "revision": "abc123",
        "prompt_text": "Transcribe the audio.",
        "prompt_file": None,
        "en_prompt_text": None,
        "en_prompt_file": None,
        "system_prompt": None,
        "system_prompt_file": None,
        "prompt_content_order": "text_audio",
        "max_output_tokens": 256,
        "vllm_kwargs": {
            "max_model_len": 32768,
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.95,
            "dtype": "auto",
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "prefix_caching_hash_algo": "xxhash",
            "limit_mm_per_prompt": {
                "image": 0,
                "video": 0,
                "audio": 2,
            },
            "seed": 1234,
        },
        "sampling_kwargs": {
            "temperature": 0.0,
            "top_k": 1,
            "repetition_penalty": 1.0,
        },
    }
    assert executor.__class__.__name__ == "RayDataExecutor"
    assert executor.config == {}


def test_qwen_asr_tutorial_yaml_uses_generic_adapter_contract():
    config_dir = Path(__file__).parents[2] / "tutorials" / "audio" / "qwen_asr"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl",
                "pred_text_key=custom_prediction",
                "model_revision=abc123",
            ],
        )

    pipeline = create_pipeline_from_yaml(cfg, log_config=False)
    resample_stage = pipeline.stages[1]
    stage = pipeline.stages[2]
    executor = create_executor_from_yaml(cfg)

    assert resample_stage.__class__.__name__ == "ResampleAudioStage"
    assert "sample_rate" not in cfg
    assert Path(resample_stage.resampled_audio_dir).parts[-2:] == ("qwen_asr_workspace", "audio_resampled")
    assert resample_stage.target_sample_rate == 16000
    assert resample_stage.target_format == "wav"
    assert resample_stage.target_nchannels == 1
    assert stage.audio_filepath_key == "resampled_audio_filepath"
    assert stage.inputs()[1] == ["resampled_audio_filepath"]
    assert stage.pred_text_key == "custom_prediction"
    assert stage.extras_key == "asr_extras"
    assert stage.outputs() == (
        [],
        ["custom_prediction", "_skipme", "additional_notes", "asr_extras"],
    )
    assert stage.default_language is None
    assert stage.supported_language_codes == [
        "zh",
        "en",
        "yue",
        "ar",
        "de",
        "fr",
        "es",
        "pt",
        "id",
        "it",
        "ko",
        "ru",
        "th",
        "vi",
        "ja",
        "tr",
        "hi",
        "ms",
        "nl",
        "sv",
        "da",
        "fi",
        "pl",
        "cs",
        "fil",
        "fa",
        "el",
        "hu",
        "mk",
        "ro",
    ]
    assert stage.batch_size == 128
    assert stage.resources.gpus == 1
    assert dict(stage.adapter_kwargs) == {
        "revision": "abc123",
        "gpu_memory_utilization": 0.7,
        "max_new_tokens": 4096,
        "max_inference_batch_size": 128,
        "vllm_kwargs": {"max_model_len": 8192},
    }

    assert executor.__class__.__name__ == "RayDataExecutor"
    assert executor.config == {}


def test_faster_whisper_tutorial_yaml_matches_reference_contract():
    config_dir = Path(__file__).parents[2] / "tutorials" / "audio" / "faster_whisper"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl",
                "pred_text_key=custom_prediction",
                "model_revision=abc123",
                "default_language=en",
            ],
        )

    pipeline = create_pipeline_from_yaml(cfg, log_config=False)
    reader, resample_stage, stage, writer = pipeline.stages
    executor = create_executor_from_yaml(cfg)

    assert reader.__class__.__name__ == "ManifestReader"
    assert resample_stage.__class__.__name__ == "ResampleAudioStage"
    assert Path(resample_stage.resampled_audio_dir).parts[-2:] == (
        "faster_whisper_workspace",
        "audio_resampled",
    )
    assert resample_stage.target_sample_rate == 16000
    assert resample_stage.target_format == "wav"
    assert resample_stage.target_nchannels == 1

    assert stage.__class__.__name__ == "ASRStage"
    assert stage.name == "FasterWhisper_inference"
    assert stage.adapter_target == "nemo_curator.models.asr.faster_whisper.FasterWhisperASR"
    assert stage.model_id == "large-v3"
    assert stage.audio_filepath_key == "resampled_audio_filepath"
    assert stage.inputs()[1] == ["resampled_audio_filepath"]
    assert stage.target_sample_rate == 16000
    assert stage.default_language == "en"
    expected_language_codes = [
        "en",
        "zh",
        "de",
        "es",
        "ru",
        "ko",
        "fr",
        "ja",
        "pt",
        "tr",
        "pl",
        "ca",
        "nl",
        "ar",
        "sv",
        "it",
        "id",
        "hi",
        "fi",
        "vi",
        "he",
        "uk",
        "el",
        "ms",
        "cs",
        "ro",
        "da",
        "hu",
        "ta",
        "no",
        "th",
        "ur",
        "hr",
        "bg",
        "lt",
        "la",
        "mi",
        "ml",
        "cy",
        "sk",
        "te",
        "fa",
        "lv",
        "bn",
        "sr",
        "az",
        "sl",
        "kn",
        "et",
        "mk",
        "br",
        "eu",
        "is",
        "hy",
        "ne",
        "mn",
        "bs",
        "kk",
        "sq",
        "sw",
        "gl",
        "mr",
        "pa",
        "si",
        "km",
        "sn",
        "yo",
        "so",
        "af",
        "oc",
        "ka",
        "be",
        "tg",
        "sd",
        "gu",
        "am",
        "yi",
        "lo",
        "uz",
        "fo",
        "ht",
        "ps",
        "tk",
        "nn",
        "mt",
        "sa",
        "lb",
        "my",
        "bo",
        "tl",
        "mg",
        "as",
        "tt",
        "haw",
        "ln",
        "ha",
        "ba",
        "jw",
        "su",
        "yue",
        "fil",
        "in",
        "iw",
        "ji",
        "jv",
        "nb",
    ]
    assert stage.supported_language_codes == expected_language_codes
    assert len(stage.supported_language_codes) == 106
    assert stage.pred_text_key == "custom_prediction"
    assert stage.extras_key == "asr_extras"
    assert stage.outputs() == (
        [],
        ["custom_prediction", "_skipme", "additional_notes", "asr_extras"],
    )
    assert stage.batch_size == 128
    assert stage.resources.gpus == 1
    assert dict(stage.adapter_kwargs) == {
        "revision": "abc123",
        "compute_type": "float16",
        "cpu_compute_type": "int8",
        "beam_size": 5,
        "vad_filter": True,
        "without_timestamps": True,
    }

    assert writer.__class__.__name__ == "ManifestWriterStage"
    assert writer.output_path == "./faster_whisper_output.jsonl"
    assert executor.__class__.__name__ == "RayDataExecutor"
    assert executor.config == {}


def test_nemo_fastconformer_tutorial_yaml_uses_shared_adapter_contract():
    config_dir = Path(__file__).parents[2] / "tutorials" / "audio" / "nemo_fastconformer"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl",
                "pred_text_key=custom_prediction",
            ],
        )

    pipeline = create_pipeline_from_yaml(cfg, log_config=False)
    reader, resample_stage, stage, writer = pipeline.stages
    executor = create_executor_from_yaml(cfg)

    assert reader.__class__.__name__ == "ManifestReader"
    assert resample_stage.__class__.__name__ == "ResampleAudioStage"
    assert resample_stage.target_sample_rate == 16000
    assert resample_stage.target_format == "wav"
    assert resample_stage.target_nchannels == 1
    assert Path(resample_stage.resampled_audio_dir).parts[-2:] == (
        "nemo_fastconformer_workspace",
        "audio_resampled",
    )
    assert stage.__class__.__name__ == "ASRStage"
    assert stage.adapter_target == "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter"
    assert stage.model_id == "nvidia/stt_en_fastconformer_ctc_large"
    assert stage.audio_filepath_key == "resampled_audio_filepath"
    assert stage.target_sample_rate == 16000
    assert stage.pred_text_key == "custom_prediction"
    assert stage.batch_size == 16
    assert stage.resources.gpus == 1
    assert dict(stage.adapter_kwargs) == {
        "num_workers": 0,
        "verbose": False,
        "enable_local_attention": False,
    }
    assert writer.__class__.__name__ == "ManifestWriterStage"
    assert executor.__class__.__name__ == "RayDataExecutor"
    assert executor.config == {}


def test_run_cli_defaults_to_pipeline_config_for_fastconformer() -> None:
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(repo_root / "nemo_curator/config/run.py"),
            "--config-path",
            "../../tutorials/audio/nemo_fastconformer",
            "--cfg",
            "job",
            "manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "adapter_target: nemo_curator.models.asr.nemo_asr.NeMoASRAdapter" in result.stdout
