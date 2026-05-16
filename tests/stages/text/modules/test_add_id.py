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

import pandas as pd
import pytest

from nemo_curator.stages.text.modules.add_id import AddId
from nemo_curator.tasks import DocumentBatch


def _sample_batch() -> DocumentBatch:
    """Create a simple three-row batch for tests."""
    df = pd.DataFrame({"text": ["first", "second", "third"]})
    return DocumentBatch(data=df, task_id="batch_1", dataset_name="test_ds")


class TestAddIdStage:
    def test_add_id_basic(self) -> None:
        """IDs are added in the expected format and other columns are untouched."""
        batch = _sample_batch()
        stage = AddId(id_field="doc_id")

        stage.setup()
        result = stage.process(batch)
        assert result is not None, "Stage returned None"

        prefix = str(batch._uuid)
        expected_ids = [f"{prefix}_{i}" for i in range(len(batch.to_pandas()))]

        # Check column creation and values
        assert list(result.data["doc_id"]) == expected_ids

        # Original data should remain unchanged
        pd.testing.assert_series_equal(batch.data["text"], result.data["text"])

        # Task id should include the stage name
        assert result.task_id == f"{batch.task_id}_{stage.name}"

    def test_io_spec(self) -> None:
        """The declared inputs/outputs match the contract."""
        stage = AddId(id_field="custom_id")
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], ["custom_id"])

    def test_unique_ids_across_batches(self) -> None:
        """Ensure IDs are unique across different batches."""
        batch1 = _sample_batch()
        batch2 = _sample_batch()

        stage = AddId(id_field="id")

        res1 = stage.process(batch1)
        res2 = stage.process(batch2)

        ids1 = set(res1.data["id"])
        ids2 = set(res2.data["id"])

        assert ids1.isdisjoint(ids2), "IDs should be unique across batches"

        # Additional sanity checks
        assert len(ids1) == len(batch1.to_pandas())
        assert len(ids2) == len(batch2.to_pandas())

    def test_id_prefix_is_applied(self) -> None:
        """IDs should be prefixed with the supplied id_prefix."""
        batch = _sample_batch()
        stage = AddId(id_field="uid", id_prefix="custom")

        result = stage.process(batch)

        prefix = f"custom_{batch._uuid}"
        expected_ids = [f"{prefix}_{i}" for i in range(len(batch.to_pandas()))]
        assert list(result.data["uid"]) == expected_ids

    def test_overwrite_false_raises_error(self) -> None:
        """If the column already exists and overwrite=False, a ValueError is raised."""
        batch = _sample_batch()
        batch.data["dup_id"] = ["x", "y", "z"]

        stage = AddId(id_field="dup_id", overwrite=False)
        # Expect the specific message from AddId.process when overwrite=False
        with pytest.raises(ValueError, match="Column 'dup_id' already exists"):
            stage.process(batch)

    def test_overwrite_true_replaces_column(self) -> None:
        """If overwrite=True, existing column values should be replaced with new IDs."""
        batch = _sample_batch()
        batch.data["my_id"] = ["old0", "old1", "old2"]

        stage = AddId(id_field="my_id", overwrite=True)
        result = stage.process(batch)

        prefix = str(batch._uuid)
        expected_ids = [f"{prefix}_{i}" for i in range(len(batch.to_pandas()))]
        assert list(result.data["my_id"]) == expected_ids
        # Ensure the old values are gone
        assert not set(result.data["my_id"]).intersection({"old0", "old1", "old2"})
