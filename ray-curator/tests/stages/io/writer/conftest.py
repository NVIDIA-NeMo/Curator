"""Shared fixtures for writer tests."""

import pandas as pd
import pytest

from ray_curator.tasks import DocumentBatch
from ray_curator.utils.performance_utils import StagePerfStats


@pytest.fixture
def pandas_document_batch() -> DocumentBatch:
    """Fixture providing a pandas DataFrame for testing."""
    df = pd.DataFrame(
        {
            "text": ["Hello pandas", "This is pandas test", "Writing with pandas"],
            "id": ["pd1", "pd2", "pd3"],
            "category": ["greeting", "test", "demo"],
            "score": [0.9, 0.8, 0.95],
        }
    )
    return DocumentBatch(
        task_id="test_pandas_batch",
        dataset_name="test_dataset",
        data=df,
        _metadata={"dummy_key": "dummy_value"},
        _stage_perf=[
            StagePerfStats(
                stage_name="test_stage",
                process_time=0.1,  # 100ms = 0.1s
                actor_idle_time=0.0,
                input_data_size_mb=0.001,  # Small test data
                num_items_processed=3,  # 3 rows in the DataFrame
            )
        ],
    )


@pytest.fixture
def pyarrow_document_batch(pandas_document_batch: DocumentBatch) -> DocumentBatch:
    """Fixture providing a pyarrow Table for testing."""
    return DocumentBatch(
        task_id="test_pyarrow_batch",
        dataset_name="test_dataset",
        data=pandas_document_batch.to_pyarrow(),
        _metadata={"dummy_key": "dummy_value"},
        _stage_perf=[
            StagePerfStats(
                stage_name="test_stage",
                process_time=0.1,  # 100ms = 0.1s
                actor_idle_time=0.0,
                input_data_size_mb=0.001,  # Small test data
                num_items_processed=3,  # 3 rows in the DataFrame
            )
        ],
    )


@pytest.fixture
def document_batch(
    request: pytest.FixtureRequest,
    pandas_document_batch: DocumentBatch,
    pyarrow_document_batch: DocumentBatch,
) -> DocumentBatch:
    """Parametrizable fixture that returns either pandas or pyarrow document batch."""
    if request.param == "pandas":
        return pandas_document_batch
    elif request.param == "pyarrow":
        return pyarrow_document_batch
    else:
        msg = f"Unknown document_batch type: {request.param}"
        raise ValueError(msg)
