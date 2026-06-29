"""WARC → InterleavedBatch extraction stage."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import pandas as pd
import pyarrow as pa
from loguru import logger

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

# Map ``extractor`` arg → pure function (record, *, min_text_chars) → rows.
# Imports are lazy so magic-html is only loaded when actually used.
EXTRACTOR_CHOICES: tuple[str, ...] = ("magic_html", "magic_traf")


# Common Crawl WARC path convention:
#   .../CC-MAIN-<snapshot>/segments/<segment>/warc/<file>.warc.gz
# Snapshot/segment come out as ``None`` for paths that don't match.
_CC_PATH_RE = re.compile(
    r"(?P<snapshot>CC-MAIN-\d{4}-\d{2})/segments/(?P<segment>[^/]+)/warc/"
)


def _parse_warc_path(path: str | None) -> tuple[str | None, str | None, str | None]:
    """Return ``(warc_file, warc_segment, snapshot)`` parsed from a WARC path.
    Any field that can't be recovered comes back as ``None``."""
    if not path:
        return None, None, None
    warc_file = os.path.basename(path)
    m = _CC_PATH_RE.search(path)
    if m:
        return warc_file, m.group("segment"), m.group("snapshot")
    return warc_file, None, None


def _resolve_extractor(name: str) -> Callable:
    if name == "magic_html":
        from nemo_curator.stages.nemotron_cc_mm.extractor_magic_html import (
            warc_html_to_interleaved_rows_magic_html,
        )
        return warc_html_to_interleaved_rows_magic_html
    if name == "magic_traf":
        from nemo_curator.stages.nemotron_cc_mm.extractor_magic_html import (
            warc_html_to_interleaved_rows_magic_traf,
        )
        return warc_html_to_interleaved_rows_magic_traf
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

        * ``"magic_html"`` — magic-html main-content extraction, then walk
          the cleaned HTML.
        * ``"magic_traf"`` — magic-html first; if it yields no content rows,
          fall back to trafilatura(output_format='html') on the raw HTML
          (default).
    min_text_chars:
        Drop text runs shorter than this many chars.
    resiliparse_text:
        Also run Curator's ``ResiliparseExtractor`` on the same HTML and
        store its joined paragraphs in the metadata row's ``text_content``
        column.  Lets downstream stages compare our DOM-walker output
        against a battle-tested text-only extractor (or use Resiliparse
        text directly as the doc's "clean text").  Default ``True``.
    """

    extractor: str = "magic_traf"
    min_text_chars: int = 1
    log_counts: bool = True
    resiliparse_text: bool = True
    # Hard cap on per-row text length (characters).  Pathological docs
    # (e.g. minified JS / base64 blobs not stripped by extraction) can
    # produce single rows of multiple MB, which then blow up downstream
    # filters that materialize the column as a numpy fixed-width Unicode
    # array (allocation = n_rows × max_len × 4 bytes → easily 10+ TiB).
    # 0 disables the cap.
    max_text_chars: int = 50_000
    # Chunk the per-WARC InterleavedBatch into sub-batches no larger than
    # this many bytes (Arrow ``nbytes``).  Keeps all rows of a sample_id
    # together.  Lets downstream image / GPU stages stream rather than
    # holding the entire WARC's image bytes at once.
    # Same idiom as Curator's InterleavedParquetReaderStage.
    max_batch_bytes: int = 256 * 1024 * 1024  # 256 MiB
    name: str = "warc_to_interleaved_extract"

    # Per-worker cached state (populated in setup).
    _resiliparse: object | None = field(default=None, init=False, repr=False)
    _stop_lists: dict | None = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["url", "warc_id", "content"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Lazily load Resiliparse + Curator's stop-list dict per worker."""
        if not self.resiliparse_text:
            return
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
                html, self._stop_lists[lang], lang
            )
            if not paragraphs:
                return None
            return "\n\n".join(paragraphs)
        except Exception:  # noqa: BLE001
            return None

    def process(self, task: DocumentBatch) -> InterleavedBatch | list[InterleavedBatch]:
        df_in = task.data
        if isinstance(df_in, pa.Table):
            df_in = df_in.to_pandas()

        extract_fn = _resolve_extractor(self.extractor)

        all_rows: list[dict] = []
        n_failed = 0
        n_resiliparse_ok = 0
        n_truncated = 0
        cap = self.max_text_chars
        for _, row in df_in.iterrows():
            warc_path = row.get("source_id") or row.get("file_name") or ""
            warc_file, warc_segment, snapshot = _parse_warc_path(warc_path)
            record = {
                "url": row["url"],
                "warc_id": row["warc_id"],
                "source_id": warc_path,
                "content": row["content"],
                # Lineage — propagated into the metadata row's source_ref JSON.
                "warc_file": warc_file,
                "warc_segment": warc_segment,
                "snapshot": snapshot,
                "extractor": self.extractor,
            }
            try:
                new_rows = extract_fn(
                    record, min_text_chars=self.min_text_chars
                )
            except Exception as e:  # noqa: BLE001
                # Defensive: never let one bad record kill the batch.
                n_failed += 1
                logger.warning(
                    f"[{self.name}] extraction failed for "
                    f"warc_id={record.get('warc_id', '?')[:36]}: "
                    f"{type(e).__name__}: {e!s:.120}"
                )
                continue

            # Optional: annotate the metadata row with a parallel Resiliparse
            # text extraction.  Position == -1 marks the metadata row.
            if self.resiliparse_text:
                rp_text = self._resiliparse_text(record["content"])
                if rp_text:
                    for r in new_rows:
                        if r.get("position") == -1:
                            r["text_content"] = rp_text
                            n_resiliparse_ok += 1
                            break

            # Cap per-row text length to bound downstream numpy upcasts.
            if cap > 0:
                for r in new_rows:
                    t = r.get("text_content")
                    if t and isinstance(t, str) and len(t) > cap:
                        r["text_content"] = t[:cap]
                        n_truncated += 1

            all_rows.extend(new_rows)

        out_table = pa.Table.from_pylist(all_rows, schema=INTERLEAVED_SCHEMA)

        # Split into sub-batches by byte budget while keeping all rows of
        # the same sample_id together.  This lets downstream image/GPU
        # stages stream — each sub-batch flows through the pipeline and
        # writes its own Parquet shard independently, instead of one
        # giant WARC-wide batch sitting in memory until the writer fires.
        splits = split_table_by_group_max_bytes(
            out_table, "sample_id", self.max_batch_bytes
        )

        if self.log_counts:
            n_in = len(df_in)
            n_rows = len(all_rows)
            # Distinct sample_ids = docs that emitted at least one row.
            n_docs_out = sum(1 for r in all_rows if r.get("position") == -1)
            # Whitespace token count over text rows — baseline of the
            # token funnel that LoggingInterleavedFilterStage continues
            # reporting for every downstream filter.
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
                f"records {n_in:>6} → docs {n_docs_out:>6}, "
                f"rows {n_rows:>7}, tokens {n_tokens:>9}  "
                f"(avg {n_rows / max(n_docs_out, 1):.1f} rows/doc)"
                f"{fail_part}{rp_part}{chunk_part}{trunc_part}"
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
                )
            )
        return batches[0] if len(batches) == 1 else batches
