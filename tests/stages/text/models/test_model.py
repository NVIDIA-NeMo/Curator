# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_COLUMN, INPUT_ID_COLUMN, SEQ_ORDER_COLUMN
from nemo_curator.tasks import DocumentBatch


class MockTorchModel(nn.Module):
    def __init__(self, output_size: int = 2, device: str = "cpu"):
        super().__init__()
        self.linear = nn.Linear(10, output_size)
        self.device_name = device

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = inputs[INPUT_ID_COLUMN].shape[0]
        seq_len = inputs[INPUT_ID_COLUMN].shape[1]
        return torch.randn(batch_size, seq_len, 2, device=self.device_name)

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_name)


class MockModelStage(ModelStage):
    def __init__(self, pred_column: str = "predictions", **kwargs):
        super().__init__(**kwargs)
        self.pred_column = pred_column
        self.model = None

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pred_column]

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self.model = MockTorchModel(device="cpu")

    def process_model_output(
        self, outputs: torch.Tensor, _model_input_batch: dict[str, torch.Tensor] | None = None
    ) -> dict[str, np.ndarray]:
        # Simple processing: take mean across sequence dimension
        processed = outputs.mean(dim=1).cpu().numpy()
        return {self.pred_column: processed}

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        df_cpu = df_cpu.drop(columns=[INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN])
        result = df_cpu.copy()
        result[self.pred_column] = None

        if collected_output:
            result[self.pred_column] = collected_output[self.pred_column].tolist()

        return result


class TestModelStage:
    @patch("nemo_curator.stages.text.models.model.snapshot_download")
    def test_setup_on_node_success(self, mock_snapshot_download: Mock):
        mock_snapshot_download.return_value = "/path/to/model"
        stage = MockModelStage(
            model_identifier="test/model",
            hf_token="test_token",  # noqa: S106
        )

        stage.setup_on_node()

        mock_snapshot_download.assert_called_once_with(
            repo_id="test/model",
            cache_dir=None,
            token="test_token",  # noqa: S106
            local_files_only=False,
        )

    @patch("nemo_curator.stages.text.models.model.snapshot_download")
    def test_setup_on_node_failure(self, mock_snapshot_download: Mock):
        mock_snapshot_download.side_effect = Exception("Download failed")
        stage = MockModelStage(model_identifier="test/model")

        with pytest.raises(RuntimeError, match="Failed to download test/model"):
            stage.setup_on_node()

    def create_sample_dataframe(self, num_rows: int = 4, include_seq_order: bool = True) -> pd.DataFrame:
        data = {
            INPUT_ID_COLUMN: [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9], [10, 11, 0, 0]],
            ATTENTION_MASK_COLUMN: [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]],
            "text": ["sample text 1", "sample text 2", "sample text 3", "sample text 4"],
        }

        if include_seq_order:
            data[SEQ_ORDER_COLUMN] = [2, 0, 3, 1]

        df = pd.DataFrame(data)
        return df.iloc[:num_rows]

    def test_yield_next_batch_single_batch(self):
        stage = MockModelStage(model_identifier="test/model", model_inference_batch_size=10)
        stage.setup()

        df = self.create_sample_dataframe(4)
        batches = list(stage.yield_next_batch(df))

        assert len(batches) == 1
        batch = batches[0]

        assert INPUT_ID_COLUMN in batch
        assert ATTENTION_MASK_COLUMN in batch
        assert isinstance(batch[INPUT_ID_COLUMN], torch.Tensor)
        assert isinstance(batch[ATTENTION_MASK_COLUMN], torch.Tensor)
        assert batch[INPUT_ID_COLUMN].shape[0] == 4

    def test_yield_next_batch_multiple_batches(self):
        stage = MockModelStage(model_identifier="test/model", model_inference_batch_size=2)
        stage.setup()

        df = self.create_sample_dataframe(4)
        batches = list(stage.yield_next_batch(df))

        assert len(batches) == 2

        # Check first batch
        assert batches[0][INPUT_ID_COLUMN].shape[0] == 2
        assert batches[0][ATTENTION_MASK_COLUMN].shape[0] == 2

        # Check second batch
        assert batches[1][INPUT_ID_COLUMN].shape[0] == 2
        assert batches[1][ATTENTION_MASK_COLUMN].shape[0] == 2

    def test_collect_outputs(self):
        stage = MockModelStage(model_identifier="test/model")

        processed_outputs = [
            {"predictions": np.array([[0.1, 0.9], [0.8, 0.2]])},
            {"predictions": np.array([[0.3, 0.7], [0.6, 0.4]])},
        ]

        result = stage.collect_outputs(processed_outputs)

        assert "predictions" in result
        assert result["predictions"].shape == (4, 2)
        np.testing.assert_array_equal(
            result["predictions"], np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        )

    def test_process_with_seq_order(self):
        stage = MockModelStage(model_identifier="test/model", has_seq_order=True, model_inference_batch_size=10)
        stage.setup()

        df = self.create_sample_dataframe(4, include_seq_order=True)
        batch = DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=df)

        result = stage.process(batch).to_pandas()

        # Check that seq_order column is removed and data is sorted
        assert SEQ_ORDER_COLUMN not in result.columns
        assert "predictions" in result.columns

        # Check that the original order is restored (based on seq_order)
        expected_texts = ["sample text 2", "sample text 4", "sample text 1", "sample text 3"]
        assert result["text"].tolist() == expected_texts

    def test_process_without_seq_order(self):
        stage = MockModelStage(model_identifier="test/model", has_seq_order=False, model_inference_batch_size=10)
        stage.setup()

        df = self.create_sample_dataframe(4, include_seq_order=False)
        batch = DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=df)

        result = stage.process(batch).to_pandas()

        # Check that seq_order column is not present and data order is preserved
        assert SEQ_ORDER_COLUMN not in result.columns
        assert "predictions" in result.columns

        # Order should be preserved as-is
        expected_texts = ["sample text 1", "sample text 2", "sample text 3", "sample text 4"]
        assert result["text"].tolist() == expected_texts

    def test_process_with_unpack_inference_batch(self):
        class UnpackModelStage(MockModelStage):
            def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
                # Create a model that expects unpacked arguments
                self.model = Mock()
                self.model.device = torch.device("cpu")
                self.model.return_value = torch.randn(2, 4, 2)

        stage = UnpackModelStage(
            model_identifier="test/model", unpack_inference_batch=True, model_inference_batch_size=10
        )
        stage.setup()

        df = self.create_sample_dataframe(2)
        batch = DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=df)

        _ = stage.process(batch)

        # Check that model was called with unpacked arguments
        assert stage.model.call_count == 1
        args, kwargs = stage.model.call_args
        assert INPUT_ID_COLUMN in kwargs
        assert ATTENTION_MASK_COLUMN in kwargs

    @patch("nemo_curator.stages.text.models.model.torch.cuda.empty_cache")
    @patch("nemo_curator.stages.text.models.model.gc.collect")
    def test_teardown(self, mock_gc_collect: Mock, mock_cuda_empty_cache: Mock):
        stage = MockModelStage(model_identifier="test/model")

        stage.teardown()

        mock_gc_collect.assert_called_once()
        mock_cuda_empty_cache.assert_called_once()
