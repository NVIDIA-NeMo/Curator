"""One-pass streaming WARC → InterleavedBatch extraction.

Replaces the two-stage flow (DocumentIterateExtractStage → WarcDocumentToInterleavedStage)
with a single stage that iterates WARC records, extracts each one, and discards
the raw HTML before reading the next record.  The old flow accumulated the
entire WARC (22K records × ~50 KB HTML each = 1-1.5 GB per WARC) into a
DataFrame before extraction began; this stage never materializes that
intermediate.

Memory impact (per actor, steady-state):
    Old:  ~10 GB peak (input DataFrame + magic-html parse + walker)
    New:  ~5-6 GB peak (just magic-html parse + walker for current record)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
from loguru import logger

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA
from nemo_curator.stages.nemotron_cc_mm.extraction import (
    EXTRACTOR_CHOICES,
    _parse_warc_path,
    _resolve_extractor,
)

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass
class WarcStreamingExtractStage(ProcessingStage[FileGroupTask, InterleavedBatch]):
    """Stream WARC records → extract → InterleavedBatch in one pass.

    Each input ``FileGroupTask`` carries one or more WARC URLs.  The stage
    opens each WARC via ``CommonCrawlWarcIterator`` (which already streams
    record-by-record), runs the resolved extractor function, optionally
    annotates the metadata row with Resiliparse text, and accumulates the
    OUTPUT rows (text + image, ~10× smaller than the input HTML).  Raw
    HTML bytes are discarded as soon as one record's rows are emitted.

    Parameters mirror ``WarcDocumentToInterleavedStage`` 1:1 so this is
    a drop-in replacement at the pipeline level.
    """

    extractor: str = "magic_traf"
    min_text_chars: int = 1
    log_counts: bool = True
    resiliparse_text: bool = True
    max_text_chars: int = 50_000
    max_batch_bytes: int = 256 * 1024 * 1024
    record_limit: int | None = None
    storage_options: dict[str, Any] | None = None
    name: str = "warc_streaming_extract"

    # Per-worker cached state.
    _resiliparse: object | None = field(default=None, init=False, repr=False)
    _stop_lists: dict | None = field(default=None, init=False, repr=False)
    _extract_fn: object | None = field(default=None, init=False, repr=False)
    _iterator: object | None = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: "WorkerMetadata | None" = None) -> None:  # noqa: ARG002
        """One-time per-worker init: resolve extractor fn, build iterator,
        lazy-load Resiliparse + stop-lists."""
        if self.extractor not in EXTRACTOR_CHOICES:
            raise ValueError(
                f"Unknown extractor {self.extractor!r}.  "
                f"Options: {', '.join(EXTRACTOR_CHOICES)}.",
            )
        self._extract_fn = _resolve_extractor(self.extractor)

        from nemo_curator.stages.text.download.common_crawl.warc_iterator import (
            CommonCrawlWarcIterator,
        )
        self._iterator = CommonCrawlWarcIterator(
            storage_options=self.storage_options,
        )

        if self.resiliparse_text:
            from nemo_curator.stages.text.download.html_extractors.resiliparse import (
                ResiliparseExtractor,
            )
            from nemo_curator.stages.text.download.html_extractors.utils import (
                get_stop_list_dict,
            )
            self._resiliparse = ResiliparseExtractor()
            self._stop_lists = get_stop_list_dict()

    def _resiliparse_text(self, html_bytes: bytes) -> str | None:
        """Run Resiliparse on the raw WARC HTML; return joined paragraphs."""
        if self._resiliparse is None or self._stop_lists is None or not html_bytes:
            return None
        from nemo_curator.stages.text.download.utils import decode_html, lang_detect
        try:
            html = decode_html(html_bytes)
            if html is None:
                return None
            lang = lang_detect(html)
            if lang not in self._stop_lists:
                return None
            paragraphs = self._resiliparse.extract_text(
                html, self._stop_lists[lang], lang,
            )
            if not paragraphs:
                return None
            return "\n\n".join(paragraphs)
        except Exception:  # noqa: BLE001
            return None

    def process(
        self, task: FileGroupTask,
    ) -> InterleavedBatch | list[InterleavedBatch]:
        all_rows: list[dict] = []
        n_records = 0
        n_failed = 0
        n_resiliparse_ok = 0
        n_truncated = 0
        cap = self.max_text_chars

        for file_path in task.data:
            warc_file, warc_segment, snapshot = _parse_warc_path(file_path)
            try:
                record_iter = self._iterator.iterate(file_path)
            except Exception as e:  # noqa: BLE001
                logger.error(f"[{self.name}] iterator open failed for {file_path}: {e}")
                continue

            for record_dict in record_iter:
                if self.record_limit is not None and n_records >= self.record_limit:
                    break
                n_records += 1
                record = {
                    **record_dict,
                    "warc_file": warc_file,
                    "warc_segment": warc_segment,
                    "snapshot": snapshot,
                    "extractor": self.extractor,
                }
                try:
                    new_rows = self._extract_fn(
                        record, min_text_chars=self.min_text_chars,
                    )
                except Exception as e:  # noqa: BLE001
                    n_failed += 1
                    logger.warning(
                        f"[{self.name}] extraction failed for "
                        f"warc_id={record.get('warc_id', '?')[:36]}: "
                        f"{type(e).__name__}: {e!s:.120}",
                    )
                    continue

                if self.resiliparse_text:
                    rp_text = self._resiliparse_text(record_dict["content"])
                    if rp_text:
                        for r in new_rows:
                            if r.get("position") == -1:
                                r["text_content"] = rp_text
                                n_resiliparse_ok += 1
                                break

                if cap:
                    for r in new_rows:
                        if (
                            r.get("modality") == "text"
                            and r.get("text_content")
                            and len(r["text_content"]) > cap
                        ):
                            r["text_content"] = r["text_content"][:cap]
                            n_truncated += 1

                all_rows.extend(new_rows)
                # Hint to free the per-record raw HTML promptly.
                record_dict["content"] = None

        if not all_rows:
            # Empty WARC — return an empty InterleavedBatch with the correct schema.
            return InterleavedBatch(
                task_id=f"{task.task_id}_{self.name}",
                dataset_name=task.dataset_name,
                data=pa.Table.from_pylist([], schema=INTERLEAVED_SCHEMA),
                _metadata=dict(task._metadata),
                _stage_perf=task._stage_perf,
            )

        out_table = pa.Table.from_pylist(all_rows, schema=INTERLEAVED_SCHEMA)
        splits = split_table_by_group_max_bytes(
            out_table, "sample_id", self.max_batch_bytes,
        )

        if self.log_counts:
            n_docs_out = sum(1 for r in all_rows if r.get("position") == -1)
            n_rows = len(all_rows)
            n_tokens = sum(
                len(r["text_content"].split())
                for r in all_rows
                if r.get("modality") == "text" and r.get("text_content")
            )
            fail_part = f"  ({n_failed} failed)" if n_failed else ""
            rp_part = (
                f"  ·  resiliparse {n_resiliparse_ok}/{n_docs_out}"
                if self.resiliparse_text else ""
            )
            chunk_part = (
                f"  ·  chunked into {len(splits)} sub-batches"
                if len(splits) > 1 else ""
            )
            trunc_part = (
                f"  ·  {n_truncated} rows truncated to {cap} chars"
                if n_truncated else ""
            )
            logger.info(
                f"[{self.name}/{self.extractor}] "
                f"records {n_records:>6} → docs {n_docs_out:>6}, "
                f"rows {n_rows:>7}, tokens {n_tokens:>9}  "
                f"(avg {n_rows / max(n_docs_out, 1):.1f} rows/doc)"
                f"{fail_part}{rp_part}{chunk_part}{trunc_part}",
            )

        batches: list[InterleavedBatch] = []
        for idx, split in enumerate(splits):
            task_id = (
                f"{task.task_id}_{self.name}"
                if len(splits) == 1
                else f"{task.task_id}_{self.name}_{idx:05d}"
            )
            batches.append(
                InterleavedBatch(
                    task_id=task_id,
                    dataset_name=task.dataset_name,
                    data=split,
                    _metadata=dict(task._metadata),
                    _stage_perf=task._stage_perf,
                ),
            )
        return batches[0] if len(batches) == 1 else batches
