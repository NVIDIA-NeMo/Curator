"""Parallel async image downloader for ``InterleavedBatch``.

For each image row whose ``binary_content`` is empty, parse the URL out of
``source_ref`` and fetch the bytes over HTTP.  Successes fill
``binary_content`` (and ``content_type``); failures leave
``binary_content`` empty and set ``materialize_error``.

Bounded concurrency, per-request timeout, optional retry, per-image size
cap.  Uses ``aiohttp`` inside a single ``asyncio.run`` per Ray Data
batch (each Ray worker handles one batch sequentially).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiohttp
import pyarrow as pa
from loguru import logger

from nemo_curator.stages.interleaved.stages import BaseInterleavedAnnotatorStage

if TYPE_CHECKING:
    import pandas as pd
    from nemo_curator.tasks import InterleavedBatch


# ---------------------------------------------------------------------------
# Async fetch helpers
# ---------------------------------------------------------------------------
async def _fetch_one(
    session: aiohttp.ClientSession,
    url: str,
    *,
    max_bytes: int,
    timeout_s: float,
) -> tuple[bytes | None, str | None, str | None]:
    """Fetch one URL.  Returns ``(bytes, content_type, error_or_None)``."""
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                return None, None, f"http_{resp.status}"
            content_type = (
                (resp.headers.get("Content-Type", "") or "")
                .split(";")[0]
                .strip()
                .lower()
            )
            buf = bytearray()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    return None, None, "oversized"
            return bytes(buf), content_type, None
    except asyncio.TimeoutError:
        return None, None, "timeout"
    except aiohttp.ClientError as e:
        return None, None, f"client_error:{type(e).__name__}"
    except Exception as e:  # noqa: BLE001
        return None, None, f"error:{type(e).__name__}"


async def _fetch_batch(
    urls: list[str],
    *,
    concurrency: int,
    max_bytes: int,
    timeout_s: float,
    user_agent: str,
    max_retries: int,
) -> list[tuple[bytes | None, str | None, str | None]]:
    """Fetch many URLs in parallel.  Returns aligned list with URL order."""
    if not urls:
        return []
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(
        limit=concurrency, ssl=False, ttl_dns_cache=300
    )
    headers = {"User-Agent": user_agent}
    results: list[tuple[bytes | None, str | None, str | None]] = [
        (None, None, "unstarted")
    ] * len(urls)

    async def _bounded(i: int, url: str) -> None:
        async with sem:
            for attempt in range(max_retries + 1):
                bytes_data, content_type, err = await _fetch_one(
                    session, url, max_bytes=max_bytes, timeout_s=timeout_s
                )
                if bytes_data is not None:
                    results[i] = (bytes_data, content_type, None)
                    return
                if attempt == max_retries:
                    results[i] = (None, None, err)
                    return
                # back-off ~250ms before retry
                await asyncio.sleep(0.25)

    async with aiohttp.ClientSession(
        connector=connector, headers=headers
    ) as session:
        await asyncio.gather(
            *(_bounded(i, url) for i, url in enumerate(urls))
        )
    return results


# ---------------------------------------------------------------------------
# Curator stage
# ---------------------------------------------------------------------------
@dataclass
class ParallelImageDownloader(BaseInterleavedAnnotatorStage):
    """Fetch image bytes for image rows lacking ``binary_content``.

    Mutates rows in place: ``binary_content`` and ``content_type`` filled
    on success; ``materialize_error`` set on failure with a short code.
    """

    concurrency: int = 500
    timeout_s: float = 20.0
    max_retries: int = 1
    max_bytes: int = 20 * 1024 * 1024  # 20 MB
    user_agent: str = "nemotron-cc-mm/0.1"
    log_stats: bool = True
    url_dedup: bool = True
    """Mirrors OmniCorpus §3.2: fetch each distinct URL once per batch and
    broadcast the result to all rows sharing it.  Catches most of the
    "Bloom filter" win without cross-worker shared state.  Scope is
    within-batch (Phase 1); cross-batch / cross-WARC is Phase 2."""
    name: str = "parallel_image_downloader"

    def annotate(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "modality" not in df.columns:
            return df

        # Pick image rows that haven't been materialized yet.
        is_image = df["modality"] == "image"
        is_empty = df["binary_content"].isna() | (
            df["binary_content"].map(
                lambda v: isinstance(v, (bytes, bytearray)) and len(v) == 0,
                na_action="ignore",
            ).fillna(False)
        )
        candidates = df[is_image & is_empty]
        if candidates.empty:
            return df

        # ---- Collect (idx, url) pairs from all candidate image rows ----
        indices: list[int] = []
        urls: list[str] = []
        for idx, row in candidates.iterrows():
            src_ref = row.get("source_ref")
            if not isinstance(src_ref, str) or not src_ref:
                continue
            try:
                url = json.loads(src_ref).get("url")
            except (TypeError, ValueError):
                continue
            if not url or not isinstance(url, str):
                continue
            indices.append(idx)
            urls.append(url)

        if not urls:
            return df

        # ---- Within-batch URL dedup (OmniCorpus §3.2 "Bloom filter") ----
        # Fetch each unique URL once, broadcast result to all rows that
        # share it.  ``url_to_fetch_idx`` maps URL → its index in the
        # ``unique_urls`` fetch list.
        if self.url_dedup:
            unique_urls: list[str] = []
            url_to_fetch_idx: dict[str, int] = {}
            for u in urls:
                if u not in url_to_fetch_idx:
                    url_to_fetch_idx[u] = len(unique_urls)
                    unique_urls.append(u)
            fetch_urls = unique_urls
        else:
            fetch_urls = urls

        fetch_results = asyncio.run(
            _fetch_batch(
                fetch_urls,
                concurrency=self.concurrency,
                max_bytes=self.max_bytes,
                timeout_s=self.timeout_s,
                user_agent=self.user_agent,
                max_retries=self.max_retries,
            )
        )

        if self.url_dedup:
            # Broadcast each unique-URL result back to all rows sharing it.
            results = [fetch_results[url_to_fetch_idx[u]] for u in urls]
            n_dedup_saved = len(urls) - len(fetch_urls)
        else:
            results = fetch_results
            n_dedup_saved = 0

        # Ensure object dtype so we can store bytes.
        if df["binary_content"].dtype != object:
            df["binary_content"] = df["binary_content"].astype(object)
        if df["content_type"].dtype != object:
            df["content_type"] = df["content_type"].astype(object)
        if df["materialize_error"].dtype != object:
            df["materialize_error"] = df["materialize_error"].astype(object)

        n_success = 0
        error_counter: dict[str, int] = {}
        for idx, (bytes_data, content_type, err) in zip(
            indices, results, strict=False
        ):
            if bytes_data is not None:
                df.at[idx, "binary_content"] = bytes_data
                if content_type:
                    df.at[idx, "content_type"] = content_type
                n_success += 1
            else:
                err_code = err or "unknown"
                df.at[idx, "materialize_error"] = err_code
                error_counter[err_code] = error_counter.get(err_code, 0) + 1

        if self.log_stats:
            total = len(results)
            n_failure = total - n_success
            pct = (n_success / total * 100.0) if total else 0.0
            err_summary = ", ".join(
                f"{k}={v}"
                for k, v in sorted(
                    error_counter.items(), key=lambda kv: -kv[1]
                )[:5]
            )
            dedup_str = (
                f"  ·  url-dedup saved {n_dedup_saved} requests "
                f"({n_dedup_saved / total * 100.0:.1f}%)"
                if self.url_dedup and total else ""
            )
            logger.info(
                f"[{self.name}] "
                f"fetched {n_success}/{total} = {pct:.1f}% success  "
                f"(failures: {n_failure}; top errors: {err_summary or '—'})"
                f"{dedup_str}"
            )

        return df
