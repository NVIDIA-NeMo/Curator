# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.io.readers.parquet import InterleavedParquetReaderStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.schema import PAGE_OUTPUT_SCHEMA, align_page_output
from nemo_curator.tasks import FileGroupTask
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import partition, project


def test_heterogeneous_document_files_can_be_read_together(tmp_path: Path):
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "outputs"),
        schema=PAGE_OUTPUT_SCHEMA,
        reject_unknown_columns=True,
        materialize_on_write=False,
        on_materialize_error="warn",
        atomic_local_write=True,
        mode="error",
    )
    paths, expected = [], []
    for name, kwargs in [
        ("ordinary", {}),
        ("text", {"text_only": {0}}),
        ("failed", {"failed": {0}}),
        ("empty", {"no_elements": {0}}),
    ]:
        _, tasks = partition(tmp_path, [{"file_name": f"{name}.pdf", "num_pages": 1}])
        document = PDFDocumentMergeStage().process_batch(project(tasks[0], **kwargs))[0]
        expected.extend(document.data.to_pylist())
        paths.extend(writer.process(document).data)
    for path in paths:
        assert pq.read_schema(path).equals(PAGE_OUTPUT_SCHEMA, check_metadata=False)
    output = InterleavedParquetReaderStage().process(FileGroupTask(dataset_name="test", data=paths))
    assert output.data.to_pylist() == expected


def test_schema_rejects_undeclared_columns_instead_of_dropping_them():
    table = pa.Table.from_pylist([], schema=PAGE_OUTPUT_SCHEMA).append_column("customer_metadata", pa.array([]))
    with pytest.raises(ValueError, match="customer_metadata"):
        align_page_output(table)
