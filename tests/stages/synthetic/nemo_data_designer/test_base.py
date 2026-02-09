"""
Unit tests for nemo_curator.stages.synthetic.nemo_data_designer.base module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.stages.synthetic.nemo_data_designer.base import BaseDataDesignerStage


def _import_base_stage() -> type[BaseDataDesignerStage]:
    """Import after patches to avoid loading real data_designer at collection time."""
    from nemo_curator.stages.synthetic.nemo_data_designer.base import BaseDataDesignerStage

    return BaseDataDesignerStage


# -----------------------------------------------------------------------------
# __post_init__ validation
# -----------------------------------------------------------------------------


def test_post_init_raises_when_both_none() -> None:
    """Either config_builder or data_designer_config_file must be set."""
    stage_cls = _import_base_stage()
    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd"), patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ), pytest.raises(ValueError, match="Either .* must be set"):
        stage_cls(
            config_builder=None,
            data_designer_config_file=None,
        )


def test_post_init_raises_when_both_set() -> None:
    """Only one of config_builder or data_designer_config_file can be set."""
    stage_cls = _import_base_stage()
    mock_builder = MagicMock()
    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd"), patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ), pytest.raises(ValueError, match="Only one of .* can be set"):
        stage_cls(
            config_builder=mock_builder,
            data_designer_config_file="/path/to/config.yaml",
        )


def test_post_init_ok_with_config_builder_only() -> None:
    """No error when only config_builder is set."""
    stage_cls = _import_base_stage()
    mock_builder = MagicMock()
    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd"), patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ):
        stage = stage_cls(config_builder=mock_builder)
    assert stage.config_builder is mock_builder
    assert stage.data_designer_config_file is None


def test_post_init_ok_with_config_file_only() -> None:
    """No error when only data_designer_config_file is set."""
    stage_cls = _import_base_stage()
    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd"), patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ):
        stage = stage_cls(data_designer_config_file="/some/config.yaml")
    assert stage.config_builder is None
    assert stage.data_designer_config_file == "/some/config.yaml"


# -----------------------------------------------------------------------------
# name, resources, inputs, outputs
# -----------------------------------------------------------------------------


def test_name() -> None:
    """Stage name is NemoDataDesignerBaseStage."""
    stage_cls = _import_base_stage()
    mock_builder = MagicMock()
    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd"), patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ):
        stage = stage_cls(config_builder=mock_builder)
    assert stage.name == "NemoDataDesignerBaseStage"


def test_resources_default_gpus_zero() -> None:
    """Resources use num_gpus_per_worker (default 0)."""
    stage_cls = _import_base_stage()
    mock_builder = MagicMock()
    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd"), patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ):
        stage = stage_cls(config_builder=mock_builder)
    assert stage.resources == Resources(gpus=0.0)


def test_resources_custom_gpus() -> None:
    """Resources reflect custom num_gpus_per_worker."""
    stage_cls = _import_base_stage()
    mock_builder = MagicMock()
    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd"), patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ):
        stage = stage_cls(config_builder=mock_builder, num_gpus_per_worker=2.0)
    assert stage.resources == Resources(gpus=2.0)


def test_inputs_outputs() -> None:
    """inputs and outputs return ('data', [])."""
    stage_cls = _import_base_stage()
    mock_builder = MagicMock()
    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd"), patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ):
        stage = stage_cls(config_builder=mock_builder)
    assert stage.inputs() == (["data"], [])
    assert stage.outputs() == (["data"], [])


# -----------------------------------------------------------------------------
# setup()
# -----------------------------------------------------------------------------


def test_setup_with_config_builder_does_not_load_file() -> None:
    """When config_builder is set, setup does not call from_config."""
    stage_cls = _import_base_stage()
    mock_builder = MagicMock()
    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd") as mock_dd, patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ) as mock_dd_cls:
        stage = stage_cls(config_builder=mock_builder)
        stage.setup()
    mock_dd.DataDesignerConfigBuilder.from_config.assert_not_called()
    assert stage.config_builder is mock_builder
    mock_dd_cls.assert_called_once()


def test_setup_with_config_file_loads_config() -> None:
    """When data_designer_config_file is set, setup loads config from file."""
    stage_cls = _import_base_stage()
    mock_loaded_builder = MagicMock()
    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd") as mock_dd, patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ):
        mock_dd.DataDesignerConfigBuilder.from_config.return_value = mock_loaded_builder
        stage = stage_cls(data_designer_config_file="/path/to/config.yaml")
        stage.setup()
    mock_dd.DataDesignerConfigBuilder.from_config.assert_called_once_with("/path/to/config.yaml")
    assert stage.config_builder is mock_loaded_builder


# -----------------------------------------------------------------------------
# process()
# -----------------------------------------------------------------------------


def test_process_sets_seed_and_calls_preview_returns_batch() -> None:
    """process sets seed dataset, calls preview, and returns DocumentBatch with result df."""
    stage_cls = _import_base_stage()
    mock_builder = MagicMock()
    input_df = pd.DataFrame([{"text": "hello"}])
    output_df = pd.DataFrame([{"text": "hello", "generated": "world"}])
    mock_preview_result = MagicMock()
    mock_preview_result.dataset = output_df

    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd") as mock_dd, patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ) as mock_dd_cls:
        mock_dd.DataFrameSeedSource = MagicMock()
        mock_designer = MagicMock()
        mock_designer.preview.return_value = mock_preview_result
        mock_dd_cls.return_value = mock_designer

        stage = stage_cls(config_builder=mock_builder)
        stage.setup()

        batch = DocumentBatch(
            data=input_df,
            dataset_name="ds1",
            task_id="task-1",
            _metadata={"k": "v"},
            _stage_perf=[],
        )
        out_batch = stage.process(batch)

    mock_builder.with_seed_dataset.assert_called_once()
    call_arg = mock_builder.with_seed_dataset.call_args[0][0]
    assert call_arg == mock_dd.DataFrameSeedSource.return_value
    mock_dd.DataFrameSeedSource.assert_called_once_with(df=input_df)

    mock_designer.preview.assert_called_once_with(mock_builder, num_records=1)

    assert isinstance(out_batch, DocumentBatch)
    assert out_batch.task_id == "task-1"
    assert out_batch.dataset_name == "ds1"
    assert out_batch.data is output_df
    assert out_batch._metadata == {"k": "v"}
    assert out_batch._stage_perf == []


def test_process_logs_metrics() -> None:
    """process logs ndd_running_time, num_input_records, num_output_records."""
    stage_cls = _import_base_stage()
    mock_builder = MagicMock()
    input_df = pd.DataFrame([{"a": 1}, {"a": 2}])
    output_df = pd.DataFrame([{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}])
    mock_preview_result = MagicMock()
    mock_preview_result.dataset = output_df

    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd") as mock_dd, patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ) as mock_dd_cls:
        mock_dd.DataFrameSeedSource = MagicMock()
        mock_designer = MagicMock()
        mock_designer.preview.return_value = mock_preview_result
        mock_dd_cls.return_value = mock_designer

        stage = stage_cls(config_builder=mock_builder)
        stage.setup()

        batch = DocumentBatch(data=input_df, dataset_name="ds", task_id="t1")
        stage.process(batch)

    assert hasattr(stage, "_custom_metrics")
    assert "ndd_running_time" in stage._custom_metrics
    assert stage._custom_metrics["num_input_records"] == 2.0
    assert stage._custom_metrics["num_output_records"] == 3.0


def test_process_empty_batch() -> None:
    """process handles empty dataframe."""
    stage_cls = _import_base_stage()
    mock_builder = MagicMock()
    input_df = pd.DataFrame()
    output_df = pd.DataFrame()

    with patch("nemo_curator.stages.synthetic.nemo_data_designer.base.dd") as mock_dd, patch(
        "nemo_curator.stages.synthetic.nemo_data_designer.base.DataDesigner"
    ) as mock_dd_cls:
        mock_dd.DataFrameSeedSource = MagicMock()
        mock_designer = MagicMock()
        mock_preview_result = MagicMock()
        mock_preview_result.dataset = output_df
        mock_designer.preview.return_value = mock_preview_result
        mock_dd_cls.return_value = mock_designer

        stage = stage_cls(config_builder=mock_builder)
        stage.setup()

        batch = DocumentBatch(data=input_df, dataset_name="ds", task_id="t1")
        out_batch = stage.process(batch)

    mock_designer.preview.assert_called_once_with(mock_builder, num_records=0)
    assert len(out_batch.data) == 0
    assert out_batch.task_id == "t1"
