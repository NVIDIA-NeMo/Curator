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

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from nemo_curator.config.run import create_pipeline_from_yaml
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
