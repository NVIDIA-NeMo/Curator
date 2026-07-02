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

"""Build the unified CC HTML/text Lance dataset.

Each Slurm array job handles one split of one snapshot:

  build_cc_lancedb.py --snapshot CC-MAIN-2025-26 \\
      --split 7 --total-splits 40 \\
      --stage-only --staging-dir /lustre/.../staging \\
      --lance-uri s3://vjawa-cc-lance/cc_all --pbss

In --stage-only mode the job writes lance fragments and saves their
metadata to staging/<snapshot_id>/split_NNN.pkl — no manifest commit yet.
A separate commit_snapshot.py job collects all 40 pkl files and issues
a single LanceDataset.commit() for the snapshot.

For quick smoke-tests omit --stage-only and the job commits immediately:

  build_cc_lancedb.py --snapshot CC-MAIN-2025-26 \\
      --download-dir /tmp/warcs --lance-uri s3://... --pbss --url-limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

from nemo_curator.backends.ray_data import RayDataExecutor  # noqa: E402
from nemo_curator.pipeline import Pipeline  # noqa: E402
from nemo_curator.stages.text.download.base.stage import DocumentDownloadExtractStage  # noqa: E402
from nemo_curator.stages.text.download.common_crawl.cc_html_extract import HtmlExtractStage  # noqa: E402
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCDownloader  # noqa: E402
from nemo_curator.stages.text.download.common_crawl.warc_iterator import CommonCrawlWarcIterator  # noqa: E402
from nemo_curator.stages.text.download.html_extractors.justext import JusTextExtractor  # noqa: E402
from nemo_curator.stages.text.download.html_extractors.resiliparse import ResiliparseExtractor  # noqa: E402
from nemo_curator.stages.text.download.html_extractors.trafilatura import TrafilaturaExtractor  # noqa: E402
from nemo_curator.stages.text.io.writer.lancedb import (  # noqa: E402
    LanceFragmentTask,
    LanceFragmentWriterStage,
    lance_commit_fragments,
)
from nemo_curator.stages.text.io.writer.utils import s3_storage_options_from_env  # noqa: E402
from nemo_curator.tasks import EmptyTask  # noqa: E402
from nemo_curator.tasks.tasks import Task  # noqa: E402,TC001

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import CC_LANCE_SCHEMA  # noqa: E402

_PBSS_ENDPOINT = "https://pdx.s8k.io"
_PBSS_WARC_BUCKET = "crawl-data"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _append_manifest(manifest_file: str | None, entry: dict) -> None:
    if not manifest_file:
        return
    entry["ts"] = datetime.now(tz=UTC).isoformat()
    Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _stage_fragments(
    tasks: list[Task],
    staging_dir: str,
    snapshot_id: str,
    split: int,
    manifest_file: str | None,
) -> None:
    """Serialize LanceFragmentTask data to a pkl file for later batch commit."""
    import pickle

    fragment_tasks = [t for t in tasks if isinstance(t, LanceFragmentTask)]
    fragments = [f for t in fragment_tasks for f in t.data]
    schema = next((t.schema for t in fragment_tasks if t.schema is not None), None)

    if not fragments:
        logger.warning(f"No fragments to stage for {snapshot_id} split {split}")
        _append_manifest(manifest_file, {"event": "split_empty", "snapshot_id": snapshot_id, "split": split})
        return

    out = Path(staging_dir) / snapshot_id / f"split_{split:03d}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"snapshot_id": snapshot_id, "split": split, "fragments": fragments, "schema": schema}, f)

    logger.info(f"Staged {len(fragments)} fragments → {out}")
    _append_manifest(
        manifest_file,
        {
            "event": "split_staged",
            "snapshot_id": snapshot_id,
            "split": split,
            "fragment_count": len(fragments),
            "staging_file": str(out),
            "job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    write_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    write_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not write_key or not write_secret:
        logger.error("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set")
        sys.exit(1)

    cc_key_id = os.environ.get("CC_PBSS_ACCESS_KEY_ID") or write_key
    cc_secret = os.environ.get("CC_PBSS_SECRET_ACCESS_KEY") or write_secret

    # Strip CC-MAIN-/CC-NEWS- prefix — MainCommonCrawlUrlGenerator expects YYYY-WW.
    snapshot_id = args.snapshot
    snapshot_week = snapshot_id.removeprefix("CC-MAIN-").removeprefix("CC-NEWS-")

    # Load URLs from a pre-generated file (no internet on compute nodes) or
    # fetch live from the CC index (requires internet — login node only).
    if args.url_file:
        with open(args.url_file) as f:
            all_urls = json.load(f)
        logger.info(f"Loaded {len(all_urls)} URLs from {args.url_file}")
    else:
        # Lazy import: __post_init__ hits index.commoncrawl.org, blocked on compute nodes.
        from nemo_curator.stages.text.download.common_crawl.url_generation import (
            MainCommonCrawlUrlGenerator,
            NewsCommonCrawlUrlGenerator,
        )

        url_gen_cls = NewsCommonCrawlUrlGenerator if args.crawl_type == "news" else MainCommonCrawlUrlGenerator
        all_urls = url_gen_cls(
            start_snapshot_str=snapshot_week, end_snapshot_str=snapshot_week, limit=None
        ).generate_urls()

    # Slice deterministically: all_urls[split::total_splits]
    split_urls = all_urls[args.split :: args.total_splits]
    if args.url_limit:
        split_urls = split_urls[: args.url_limit]
    logger.info(f"Split {args.split}/{args.total_splits}: {len(split_urls)} WARCs")

    # Wrap the pre-sliced list so DocumentDownloadExtractStage can consume it
    # without triggering the CC index HTTP call.
    from nemo_curator.stages.text.download.base.url_generation import URLGenerator

    class _StaticURLGenerator(URLGenerator):
        def generate_urls(self) -> list[str]:  # type: ignore[override]
            return list(split_urls)  # copy so callers cannot mutate the shared slice

    url_generator: URLGenerator = _StaticURLGenerator()

    downloader = CommonCrawlWARCDownloader(
        download_dir=args.download_dir,
        use_aws_to_download=args.pbss,
        s3_bucket=_PBSS_WARC_BUCKET if args.pbss else "commoncrawl",
        s3_endpoint_url=_PBSS_ENDPOINT if args.pbss else None,
        s3_key_id=cc_key_id if args.pbss else None,
        s3_secret=cc_secret if args.pbss else None,
    )

    lance_storage_options = s3_storage_options_from_env()

    pipeline = Pipeline(
        name="cc_lance",
        stages=[
            DocumentDownloadExtractStage(
                url_generator=url_generator,
                downloader=downloader,
                iterator=CommonCrawlWarcIterator(snapshot_id=snapshot_id),
                extractor=None,
                url_limit=None,  # already sliced above
                add_filename_column=False,  # source_id already carries the WARC filename
            ),
            # All 3 extractors in one actor — eliminates 2 intermediate object-store
            # queues that caused backpressure when stages ran as separate actor pools.
            HtmlExtractStage(
                [TrafilaturaExtractor, JusTextExtractor, ResiliparseExtractor],
                ["cc_extracted_text_trafilatura", "cc_extracted_text_justext", "cc_extracted_text_resiliparse"],
                name="html_extract_all",
            ),
            LanceFragmentWriterStage(
                uri=args.lance_uri,
                schema=CC_LANCE_SCHEMA,
                storage_options=lance_storage_options,
            ),
        ],
    )

    # Cap Ray to the CPUs Slurm actually allocated (SLURM_CPUS_PER_TASK).
    # Without this, Ray auto-detects the full node (64 CPUs) and tries to schedule
    # more actors than the job owns, causing "Failed to get allocated resources".
    import ray as _ray

    _ray.init(
        num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", "32")),
        _temp_dir=os.environ["RAY_TMPDIR"],
        ignore_reinit_error=True,
    )
    tasks = pipeline.run(RayDataExecutor(), initial_tasks=[EmptyTask(dataset_name="cc_lance")])

    if args.stage_only:
        _stage_fragments(tasks, args.staging_dir, snapshot_id, args.split, args.manifest_file)
    else:
        lance_commit_fragments(tasks, uri=args.lance_uri, storage_options=lance_storage_options)
        _append_manifest(
            args.manifest_file,
            {
                "event": "split_committed",
                "snapshot_id": snapshot_id,
                "split": args.split,
                "job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build unified CC Lance dataset (one split of one snapshot).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snapshot", required=True, help="CC snapshot (e.g. CC-MAIN-2025-26).")
    parser.add_argument("--download-dir", required=True, help="Local directory for WARC downloads.")
    parser.add_argument("--lance-uri", required=True, help="Lance dataset URI (e.g. s3://bucket/cc_all).")
    parser.add_argument("--pbss", action="store_true", help="Use PBSS mirror instead of public CC.")
    parser.add_argument("--crawl-type", choices=["main", "news"], default="main")
    parser.add_argument("--url-limit", type=int, default=None, help="Max WARCs per split (testing).")
    parser.add_argument(
        "--url-file",
        default=None,
        help="JSON file of pre-generated WARC URLs (avoids internet call on compute nodes).",
    )
    # Split args
    parser.add_argument("--split", type=int, default=0, help="This job's split index (0-based).")
    parser.add_argument("--total-splits", type=int, default=1, help="Total number of parallel splits.")
    # Stage-only mode
    parser.add_argument(
        "--stage-only", action="store_true", help="Write fragments and save metadata to --staging-dir; do not commit."
    )
    parser.add_argument(
        "--staging-dir",
        default="/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_staging",
        help="Directory for staged fragment metadata files.",
    )
    parser.add_argument("--manifest-file", default=None, help="JSONL file to append progress entries to.")

    main(parser.parse_args())
