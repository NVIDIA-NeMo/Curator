"""WARC → InterleavedBatch extraction stage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
import pyarrow as pa
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

# Map ``extractor`` arg → pure function (record, *, min_text_chars) → rows.
# Imports are lazy so magic-html is only loaded when actually used.
EXTRACTOR_CHOICES: tuple[str, ...] = ("naive", "magic_html", "hybrid")


def _resolve_extractor(name: str) -> Callable:
    if name == "naive":
        from nemo_curator.stages.nemotron_cc_mm.extractor import (
            warc_html_to_interleaved_rows,
        )
        return warc_html_to_interleaved_rows
    if name == "magic_html":
        from nemo_curator.stages.nemotron_cc_mm.extractor_magic_html import (
            warc_html_to_interleaved_rows_magic_html,
        )
        return warc_html_to_interleaved_rows_magic_html
    if name == "hybrid":
        from nemo_curator.stages.nemotron_cc_mm.extractor_magic_html import (
            warc_html_to_interleaved_rows_hybrid,
        )
        return warc_html_to_interleaved_rows_hybrid
    raise ValueError(
        f"Unknown extractor {name!r}.  Options: {', '.join(EXTRACTOR_CHOICES)}."
    )


@dataclass
class WarcDocumentToInterleavedStage(
    ProcessingStage[DocumentBatch, InterleavedBatch]
):
    """Convert WARC ``DocumentBatch`` records to ``InterleavedBatch`` rows.

    Input columns (from ``CommonCrawlWarcIterator``):
        ``url`` (str), ``warc_id`` (str), ``content`` (bytes),
        optionally ``source_id`` or ``file_name``.

    Output is an ``InterleavedBatch`` whose internal table follows
    ``INTERLEAVED_SCHEMA`` — one document becomes one metadata row plus
    a sequence of content rows preserving native DOM order.

    Parameters
    ----------
    extractor:
        Which HTML → rows implementation to use.  One of:

        * ``"naive"``      — bs4 DOM walker over raw HTML (default).
        * ``"magic_html"`` — magic-html main-content extraction, then walk
          the cleaned HTML.
        * ``"hybrid"``     — magic-html first; if it yields no content rows,
          fall back to the naive walker on the raw HTML.
    min_text_chars:
        Drop text runs shorter than this many chars.
    """

    extractor: str = "naive"
    min_text_chars: int = 1
    log_counts: bool = True
    name: str = "warc_to_interleaved_extract"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["url", "warc_id", "content"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> InterleavedBatch:
        df_in = task.data
        if isinstance(df_in, pa.Table):
            df_in = df_in.to_pandas()

        extract_fn = _resolve_extractor(self.extractor)

        all_rows: list[dict] = []
        n_failed = 0
        for _, row in df_in.iterrows():
            record = {
                "url": row["url"],
                "warc_id": row["warc_id"],
                "source_id": row.get("source_id") or row.get("file_name", ""),
                "content": row["content"],
            }
            try:
                all_rows.extend(
                    extract_fn(record, min_text_chars=self.min_text_chars)
                )
            except Exception as e:  # noqa: BLE001
                # Defensive: never let one bad record kill the batch.
                n_failed += 1
                logger.warning(
                    f"[{self.name}] extraction failed for "
                    f"warc_id={record.get('warc_id', '?')[:36]}: "
                    f"{type(e).__name__}: {e!s:.120}"
                )

        out_table = pa.Table.from_pylist(all_rows, schema=INTERLEAVED_SCHEMA)

        if self.log_counts:
            n_in = len(df_in)
            n_rows = len(all_rows)
            # Distinct sample_ids = docs that emitted at least one row.
            n_docs_out = sum(1 for r in all_rows if r.get("position") == -1)
            fail_part = f"  ({n_failed} failed)" if n_failed else ""
            logger.info(
                f"[{self.name}/{self.extractor}] "
                f"records {n_in:>6} → docs {n_docs_out:>6}, "
                f"rows {n_rows:>7}  (avg {n_rows / max(n_docs_out, 1):.1f} rows/doc)"
                f"{fail_part}"
            )

        return InterleavedBatch(
            task_id=f"{task.task_id}_{self.name}",
            dataset_name=task.dataset_name,
            data=out_table,
            _metadata=dict(task._metadata),
            _stage_perf=task._stage_perf,
        )
