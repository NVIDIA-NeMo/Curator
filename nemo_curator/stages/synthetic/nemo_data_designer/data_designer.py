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

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    import data_designer.config as dd
    from data_designer.interface import DataDesigner


@dataclass
class DataDesignerStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Data Designer stage.

    This class provides a Data Designer stage.
    To request GPUs, use: DataDesignerStage(...).with_(resources=Resources(gpus=X)).

    When ``verbose`` is False (default), NeMo Data Designer (NDD) log output is suppressed
    (e.g. "Preview generation in progress", "Preview complete!") so the stage is less verbose.
    Set ``verbose=True`` to see full NDD logging.

    Optional ``model_providers``: pass a list of :class:`data_designer.config.models.ModelProvider`
    to use custom or test endpoints (e.g. a mock LLM server). If None, the default DataDesigner
    providers are used.
    """

    config_builder: dd.DataDesignerConfigBuilder | None = None
    data_designer_config_file: str | None = None
    model_providers: list | None = None
    verbose: bool = False
    data_designer: DataDesigner = field(init=False)

    def __post_init__(self) -> None:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner

        # Set in __post_init__ so they are not constructor args; use .with_(resources=..., name=...) to customize.
        self.resources = Resources(gpus=0.0)
        self.name = "DataDesignerStage"

        # check config_builder and data_designer_config_file
        if self.config_builder is None and self.data_designer_config_file is None:
            msg = "Either 'config_builder' or 'data_designer_config_file' must be set."
            raise ValueError(msg)
        if self.config_builder is not None and self.data_designer_config_file is not None:
            msg = "Only one of 'config_builder' or 'data_designer_config_file' can be set, not both."
            raise ValueError(msg)

        # read config from file if config_builder is not set
        if self.config_builder is None:
            self.config_builder = dd.DataDesignerConfigBuilder.from_config(self.data_designer_config_file)
        if self.model_providers is not None:
            self.data_designer = DataDesigner(model_providers=self.model_providers)
        else:
            self.data_designer = DataDesigner()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        import data_designer.config as dd

        num_input_records = batch.num_items
        # set seed dataframe from batch
        self.config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=batch.to_pandas()))

        # When verbose is False, suppress NDD's logging (it logs "Preview generation in progress", etc.)
        ndd_logger = logging.getLogger("data_designer")
        if not self.verbose:
            _old_ndd_level = ndd_logger.level
            ndd_logger.setLevel(logging.WARNING)

        try:
            t1 = time.perf_counter()
            results = self.data_designer.preview(self.config_builder, num_records=num_input_records)
            df = results.dataset
            ndd_running_time = time.perf_counter() - t1
        finally:
            if not self.verbose:
                ndd_logger.setLevel(_old_ndd_level)

        num_output_records = len(df)

        # Token metrics from NDD stats analysis
        # (these stats are available for LLM columns only)
        output_medians = []
        input_medians = []
        if results.analysis:
            # Loop through all columns in the analysis that has LLM token stats
            for col_stat in results.analysis.column_statistics:
                in_median = getattr(col_stat, "input_tokens_median", None)
                out_median = getattr(col_stat, "output_tokens_median", None)
                if isinstance(in_median, (int, float)):
                    input_medians.append(float(in_median))
                if isinstance(out_median, (int, float)):
                    output_medians.append(float(out_median))
        # Sum across all columns that have LLM token stats
        output_tokens_median_per_record = sum(output_medians) if output_medians else 0.0
        input_tokens_median_per_record = sum(input_medians) if input_medians else 0.0

        self._log_metrics(
            {
                "ndd_running_time": ndd_running_time,
                "num_input_records": float(num_input_records),
                "num_output_records": float(num_output_records),
                "input_tokens_median_per_record": float(input_tokens_median_per_record),
                "output_tokens_median_per_record": float(output_tokens_median_per_record),
            }
        )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

# Explicitly export the class
__all__ = ["DataDesignerStage"]
