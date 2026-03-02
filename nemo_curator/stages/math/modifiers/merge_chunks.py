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

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


class ChunkMergeStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Merges chunked documents back into one row per document.

    After LLM cleanup, the pipeline has multiple rows per document (one per
    chunk). This stage deduplicates, filters invalid chunks, sorts by chunk
    order, and concatenates text back into a single row per document.
    """

    def __init__(  # noqa: PLR0913
        self,
        text_field: str = "cleaned_text",
        raw_text_field: str | None = "text",
        chunk_id_field: str = "chunk_id",
        groupby_columns: list[str] | None = None,
        no_content_markers: list[str] | None = None,
        sum_columns: list[str] | None = None,
        max_text_length: int = 900_000,
        separator: str = "\n",
    ):
        self.text_field = text_field
        self.raw_text_field = raw_text_field
        self.chunk_id_field = chunk_id_field
        self.groupby_columns = groupby_columns or ["url"]
        self.no_content_markers = no_content_markers or [
            "NO USEFUL CONTENT",
            '"NO USEFUL CONTENT"',
        ]
        self.sum_columns = sum_columns or ["num_generated_tokens", "num_input_tokens"]
        self.max_text_length = max_text_length
        self.separator = separator
        self.name = "chunk_merge"

    def inputs(self) -> tuple[list[str], list[str]]:
        required_cols = [self.text_field, self.chunk_id_field, *self.groupby_columns]
        if self.raw_text_field:
            required_cols.append(self.raw_text_field)
        return ["data"], required_cols

    def outputs(self) -> tuple[list[str], list[str]]:
        output_cols = [self.text_field, *self.groupby_columns]
        if self.raw_text_field:
            output_cols.append(self.raw_text_field)
        return ["data"], output_cols

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Merge chunked rows back into one row per document."""
        df = batch.to_pandas()

        if df.empty:
            return DocumentBatch(
                task_id=batch.task_id,
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )

        rows_before = len(df)

        # Deduplicate by (groupby_columns + chunk_id)
        dedup_cols = [*self.groupby_columns, self.chunk_id_field]
        df = df.drop_duplicates(subset=dedup_cols, keep="first")

        # Filter rows where text matches no-content markers, is null, empty, or newline
        df = df[~df[self.text_field].isin(self.no_content_markers)]
        df = df[df[self.text_field].notna()]
        df = df[~df[self.text_field].isin(["", "\n"])]

        if df.empty:
            logger.info(f"All {rows_before} rows filtered out during chunk merge")
            return DocumentBatch(
                task_id=batch.task_id,
                dataset_name=batch.dataset_name,
                data=pd.DataFrame(columns=df.columns),
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )

        # Sort by groupby_columns + chunk_id for correct ordering
        sort_cols = [*self.groupby_columns, self.chunk_id_field]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Build aggregation: concat text fields, sum token counts, first() for metadata
        agg_dict: dict[str, tuple[str, str]] = {}
        text_fields_to_concat = [self.text_field]
        if self.raw_text_field and self.raw_text_field in df.columns:
            text_fields_to_concat.append(self.raw_text_field)

        sum_cols_present = [c for c in self.sum_columns if c in df.columns]

        for col in df.columns:
            if col in self.groupby_columns:
                continue
            if col in text_fields_to_concat:
                agg_dict[col] = (col, lambda x, _sep=self.separator: _sep.join(x.astype(str)))
            elif col in sum_cols_present:
                agg_dict[col] = (col, "sum")
            else:
                agg_dict[col] = (col, "first")

        merged = df.groupby(self.groupby_columns, sort=False).agg(**agg_dict).reset_index()

        # Post-filter: remove null, empty, or newline-only merged text, and rows exceeding max_text_length
        merged = merged[
            merged[self.text_field].notna()
            & (merged[self.text_field] != "")
            & (merged[self.text_field] != "\n")
            & (merged[self.text_field].str.len() <= self.max_text_length)
        ]

        logger.info(f"Chunk merge: {rows_before} rows -> {len(merged)} documents")

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=merged.reset_index(drop=True),
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
