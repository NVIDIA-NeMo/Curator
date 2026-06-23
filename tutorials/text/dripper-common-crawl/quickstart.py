#!/usr/bin/env python3
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

"""Dripper quickstart: DripperHTMLWorkflow on 20 synthetic pages. Use --dry-run for no-GPU mode."""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from loguru import logger


def _make_synthetic_df(n: int = 20) -> pd.DataFrame:
    templates = [
        "<html><body><h1>{t}</h1><p>{b}</p></body></html>",
        "<html><body><article><h2>{t}</h2><p>{b}</p></article></body></html>",
        "<html><body><div class='post'><h3>{t}</h3><p>{b}</p></div></body></html>",
    ]
    bodies = [
        "The quick brown fox jumps over the lazy dog.",
        "Scientists discover a new method to improve efficiency.",
        "Community gathers to celebrate the annual harvest festival.",
        "Regular exercise improves cognitive function, study finds.",
        "Markets close higher on strong earnings reports this quarter.",
    ]
    rows = []
    for i in range(n):
        t, b = f"Article {i}", bodies[i % len(bodies)]
        rows.append(
            {
                "url": f"https://example{i % 3}.com/page-{i:04d}",
                "url_host_name": f"example{i % 3}.com",
                "html": templates[i % len(templates)].format(t=t, b=b),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dripper quickstart — DripperHTMLWorkflow on synthetic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server-url", default="http://localhost:8000/v1", help="Base URL of an OpenAI-compatible inference server."
    )
    parser.add_argument(
        "--model-name",
        default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact",
        help="Model ID served at --server-url.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM inference (no server needed).")
    args = parser.parse_args()

    try:
        from nemo_curator.backends.xenna import XennaExecutor
        from nemo_curator.models.client.openai_client import OpenAIClient
        from nemo_curator.stages.text.experimental.dripper import DripperHTMLWorkflow
        from nemo_curator.tasks import DocumentBatch
    except ImportError as exc:
        logger.error("Run: pip install 'nemo-curator[dripper]'\n  {}", exc)
        sys.exit(1)

    # Build the LLM client (or a no-op stub for --dry-run)
    if args.dry_run:
        from nemo_curator.models.client.llm_client import AsyncLLMClient

        class _DryRunClient(AsyncLLMClient):
            def __init__(self):
                super().__init__(max_concurrent_requests=1, max_retries=0, base_delay=0.0)

            def setup(self):
                pass

            async def _query_model_impl(
                self, *, messages, model, conversation_formatter=None, generation_config=None
            ) -> list[str]:
                return [""]

        client = _DryRunClient()
        logger.info("Dry-run mode: LLM inference skipped.")
    else:
        client = OpenAIClient(model=args.model_name, base_url=args.server_url, api_key="EMPTY")
        logger.info("Using OpenAI-compatible client at {}", args.server_url)

    # Construct the workflow
    workflow = DripperHTMLWorkflow(
        client=client,
        model_name=args.model_name,
        perform_layout_clustering=True,
        layout_cluster_threshold=0.95,
        fallback="trafilatura",
        output_format="mm_md",
    )

    # Build input tasks from a 20-row in-memory DataFrame
    df = _make_synthetic_df(n=20)
    initial_tasks = [DocumentBatch(task_id="quickstart-0", dataset_name="synthetic", data=df)]
    logger.info("Running DripperHTMLWorkflow on {} synthetic pages...", len(df))

    # Run
    result = workflow.run(executor=XennaExecutor(), initial_tasks=initial_tasks)

    # Show results
    output_tasks = result.pipeline_tasks.get("dripper_html_extraction") or []
    if output_tasks:
        out_df = output_tasks[0].to_pandas()
        sample_cols = [c for c in ["url", "dripper_content", "dripper_error"] if c in out_df.columns]
        print(out_df[sample_cols].head(5).to_string())
    else:
        logger.warning("No output tasks returned — check your pipeline configuration.")


if __name__ == "__main__":
    main()
