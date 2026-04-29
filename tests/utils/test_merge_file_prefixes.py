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

from __future__ import annotations

import os
import struct
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nemo_curator.stages.text.io.writer.megatron_tokenizer import _INDEX_HEADER, MegatronTokenizerWriter
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.merge_file_prefixes import merge_file_prefixes

# Fixed-size regions of the .idx file, see MegatronTokenizerWriter.write_idx_data
_IDX_FIXED_HEADER_BYTES = 9 + 8 + 1 + 8 + 8  # header + version + dtype code + seq count + doc count
_IDX_PER_SEQUENCE_BYTES = 4 + 8  # int32 length + int64 pointer
_IDX_PER_DOCUMENT_BYTES = 8  # int64 document index


class MockTokenizerOutput:
    def __init__(self, input_ids: list[list[int]], attention_mask: list[list[int]]) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask


@pytest.fixture
def mock_tokenizer() -> Mock:
    tokenizer = Mock()
    tokenizer.vocab_size = 2**12
    tokenizer.eos_token_id = 1

    def mock_batch_encode_plus(texts: list[str], **kwargs: Any) -> MockTokenizerOutput:  # noqa: ANN401 ARG001
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for text in texts:
            token_count = len(text.split())
            input_ids.append([*range(1000, 1000 + token_count)])
            attention_masks.append([1] * token_count)
        return MockTokenizerOutput(input_ids, attention_masks)

    tokenizer.side_effect = mock_batch_encode_plus
    return tokenizer


@pytest.fixture(autouse=True)
def setup_mocks(mock_tokenizer: Mock):
    with patch("nemo_curator.stages.text.io.writer.megatron_tokenizer.AutoTokenizer") as mock_auto_tokenizer:
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        yield {"auto_tokenizer": mock_auto_tokenizer}


def _make_batch(task_id: str, texts: list[str]) -> DocumentBatch:
    df = pd.DataFrame({"text": texts, "id": [f"{task_id}_{i}" for i in range(len(texts))]})
    return DocumentBatch(
        task_id=task_id,
        dataset_name="test_dataset",
        data=df,
        _metadata={"source_files": [f"{task_id}.jsonl"]},
    )


def _expected_idx_size(total_sequences: int) -> int:
    """Bytes a merged .idx should occupy given the total number of sequences.

    The merged file has sequence_count = total_sequences and document_count = total_sequences + 1,
    because the writer emits arange(N+1) per prefix and IndexedDatasetBuilder seeds document_indices
    with a single 0 and appends (offset + doc_indices)[1:] for each subsequent prefix.
    """
    return (
        _IDX_FIXED_HEADER_BYTES
        + total_sequences * _IDX_PER_SEQUENCE_BYTES
        + (total_sequences + 1) * _IDX_PER_DOCUMENT_BYTES
    )


class TestMergeFilePrefixes:
    """End-to-end tests covering merge_file_prefixes against MegatronTokenizerWriter output."""

    @pytest.mark.parametrize("num_batches", [2, 3, 5])
    @pytest.mark.parametrize("large_vocab_size", [True, False])
    @pytest.mark.parametrize("append_eod", [True, False])
    def test_merge_file_prefixes_produces_expected_sizes(
        self,
        num_batches: int,
        large_vocab_size: bool,
        append_eod: bool,
        tmpdir: str,
        setup_mocks: dict[str, Mock],
    ):
        """Writer output merged by merge_file_prefixes should satisfy the Megatron .bin/.idx size contract."""
        if large_vocab_size:
            setup_mocks["auto_tokenizer"].from_pretrained.return_value.vocab_size = 2**17

        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir, exist_ok=True)

        # Build batches with varying per-document token counts so sequence_lengths differ.
        batches = [
            _make_batch(
                task_id=f"batch_{i}",
                texts=[f"batch {i} doc {j} " + ("word " * (j + i + 1)) for j in range(3)],
            )
            for i in range(num_batches)
        ]

        writer = MegatronTokenizerWriter(
            path=input_dir,
            model_identifier="test/model",
            append_eod=append_eod,
        )
        writer.setup()
        results = [writer.process(b) for b in batches]

        bin_paths = [r.data[0] for r in results]
        idx_paths = [r.data[1] for r in results]
        assert len(set(bin_paths)) == num_batches, "Each batch should produce a distinct .bin path"
        assert len(set(idx_paths)) == num_batches, "Each batch should produce a distinct .idx path"
        for path in bin_paths + idx_paths:
            assert os.path.exists(path), f"Writer output missing: {path}"

        total_sequences = sum(b.num_items for b in batches)
        total_num_tokens = sum(r._metadata["num_tokens"] for r in results)
        token_size = 4 if large_vocab_size else 2
        expected_bin_size = sum(os.path.getsize(p) for p in bin_paths)

        output_prefix = os.path.join(tmpdir, "merged")
        merge_file_prefixes(input_dir, output_prefix)

        merged_bin = output_prefix + ".bin"
        merged_idx = output_prefix + ".idx"
        assert os.path.exists(merged_bin)
        assert os.path.exists(merged_idx)

        # .bin is a verbatim concatenation of the individual .bin files.
        assert os.path.getsize(merged_bin) == expected_bin_size, (
            f"Merged .bin size mismatch: expected {expected_bin_size}, got {os.path.getsize(merged_bin)}"
        )
        assert os.path.getsize(merged_bin) == total_num_tokens * token_size, (
            f"Merged .bin byte count should equal total_tokens * token_size "
            f"({total_num_tokens} * {token_size} = {total_num_tokens * token_size}), "
            f"got {os.path.getsize(merged_bin)}"
        )

        # .idx layout is fixed by MegatronTokenizerWriter.write_idx_data — 42 + 20 * N_total.
        expected_idx_size = _expected_idx_size(total_sequences)
        assert os.path.getsize(merged_idx) == expected_idx_size, (
            f"Merged .idx size mismatch: expected {expected_idx_size}, got {os.path.getsize(merged_idx)}"
        )

        # And the header/counters inside the merged .idx must line up.
        with open(merged_idx, "rb") as f:
            header = f.read(9)
            assert header == _INDEX_HEADER, f"bad header: expected {_INDEX_HEADER!r}, got {header!r}"
            version = struct.unpack("<Q", f.read(8))[0]
            assert version == 1, f"bad version: expected 1, got {version}"
            dtype_code = struct.unpack("<B", f.read(1))[0]
            expected_dtype_code = 4 if large_vocab_size else 8
            assert dtype_code == expected_dtype_code, (
                f"bad dtype code: expected {expected_dtype_code}, got {dtype_code}"
            )
            sequence_count = struct.unpack("<Q", f.read(8))[0]
            assert sequence_count == total_sequences, (
                f"bad sequence count: expected {total_sequences}, got {sequence_count}"
            )
            document_count = struct.unpack("<Q", f.read(8))[0]
            assert document_count == total_sequences + 1, (
                f"bad document count: expected {total_sequences + 1}, got {document_count}"
            )

    def test_merge_file_prefixes_single_prefix(self, tmpdir: str):
        """Merging a single prefix should reproduce the input sizes byte-for-byte."""
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir, exist_ok=True)

        batch = _make_batch(
            task_id="only_batch",
            texts=["hello world", "this is a longer test document", "tiny"],
        )

        writer = MegatronTokenizerWriter(path=input_dir, model_identifier="test/model")
        writer.setup()
        result = writer.process(batch)

        original_bin_size = os.path.getsize(result.data[0])
        original_idx_size = os.path.getsize(result.data[1])

        output_prefix = os.path.join(tmpdir, "merged")
        merge_file_prefixes(input_dir, output_prefix)

        merged_bin = output_prefix + ".bin"
        merged_idx = output_prefix + ".idx"
        assert os.path.getsize(merged_bin) == original_bin_size
        assert os.path.getsize(merged_idx) == original_idx_size
        assert os.path.getsize(merged_idx) == _expected_idx_size(batch.num_items)

    def test_merge_file_prefixes_empty_dir_raises(self, tmpdir: str):
        """An input directory without any .bin/.idx pairs should raise ValueError."""
        input_dir = os.path.join(tmpdir, "empty")
        os.makedirs(input_dir, exist_ok=True)

        output_prefix = os.path.join(tmpdir, "merged")
        with pytest.raises(ValueError, match="No valid file prefix pairs found"):
            merge_file_prefixes(input_dir, output_prefix)

    def test_merge_file_prefixes_missing_pair_raises(self, tmpdir: str):
        """A stray .bin without a matching .idx should trigger the pair-check assertion."""
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir, exist_ok=True)

        # A valid pair to prove the guard is checking each prefix, not just the directory as a whole.
        batch = _make_batch(task_id="valid", texts=["some text here", "another doc"])
        writer = MegatronTokenizerWriter(path=input_dir, model_identifier="test/model")
        writer.setup()
        writer.process(batch)

        # Orphan .bin with no corresponding .idx.
        orphan_path = os.path.join(input_dir, "orphan.bin")
        with open(orphan_path, "wb") as f:
            f.write(b"\x00" * 8)

        output_prefix = os.path.join(tmpdir, "merged")
        with pytest.raises(AssertionError, match=r"\.idx"):
            merge_file_prefixes(input_dir, output_prefix)
