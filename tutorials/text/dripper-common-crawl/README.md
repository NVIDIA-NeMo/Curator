# Dripper Common Crawl Smoke

This tutorial runs Dripper/MinerU-HTML through NeMo Curator's inference server
path on a bounded Common Crawl sample. It is intended for single-node H100
smoke runs before scaling to a full snapshot.

The Python runner:

1. Streams WARC records from `CC-MAIN-2025-26`.
2. Starts Ray through Curator's `SlurmRayClient` on SLURM, or `RayClient`
   outside SLURM.
3. Starts a Curator `InferenceServer` with the Dripper model.
4. Points `AsyncOpenAIClient` at the server endpoint.
5. Optionally runs warmup pages, then runs `DripperHTMLExtractionStage`.
6. Writes extracted rows plus steady-state and end-to-end H100-hour metrics.

Run the standalone baseline directly (single node, 8 GPUs):

```bash
python tutorials/text/dripper-common-crawl/run_mineru_html_standalone.py \
  --input-manifest-path /path/to/manifest.parquet \
  --output-dir /path/to/output --replicas 8 --max-concurrent-requests 64
```

Useful overrides: `--max-pages`, `--replicas`, `--max-concurrent-requests`,
`--warmup-pages`. Wrap this in your scheduler's job script (e.g. an `sbatch`
wrapper) for your cluster.

Throughput knobs that should not change Dripper extraction semantics:

- `ENABLE_PREFIX_CACHING=1` is the default and reuses identical prompt prefixes
  in vLLM.
- `DISABLE_THINKING=1` is the default and passes
  `chat_template_kwargs={"enable_thinking": false, "thinking": false}` through
  the OpenAI-compatible vLLM request. Dripper expects JSON/compact labels, so
  disabling thinking avoids `<think>...` text that MinerU-HTML cannot parse.
- `MAX_CONCURRENT_REQUESTS`, `MAX_NUM_SEQS`, and `MAX_NUM_BATCHED_TOKENS` tune
  request batching.
- `GPU_MEMORY_UTILIZATION` defaults to `0.9` in the Nebius wrapper to increase
  KV-cache capacity.
- `WARMUP_PAGES` excludes cold first-request overhead from the steady-state
  `h100_hours_per_page` metric while still reporting end-to-end timing.

Use `ENFORCE_EAGER=1` for short debug runs where startup time matters more than
steady-state throughput. Leave it unset for cost estimation runs.

The submit script expects PBSS/Common Crawl credentials to be available from
the environment or from the user's remote cache environment file. It does not
print secret values.
