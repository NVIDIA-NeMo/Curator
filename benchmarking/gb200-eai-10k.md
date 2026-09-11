# GB200 EAI 10k PDF sweep

Use `gb200-eai-10k.yaml` directly with `benchmarking/run.py`. Each full entry
uses four independent one-GPU Dynamo replicas on one GB200 node. Run different
entries on separate nodes with the same `--session-name`; each entry writes its
own results, GPU samples, and archived Ray/Dynamo logs. Per-entry `results.json`
includes its environment because the shared session-level `env.json` describes
the most recent invocation.

Use `/opt/venv` for the driver, adding the locked `cv2` extra on the worker:
`UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --inexact --extra cv2`, then
`source /opt/venv/bin/activate`. The image already includes benchmark tooling;
`--inexact` retains it. Use `/opt/dynamo-pdf/bin/python` only for serving actors:
it supplies OpenCV, vLLM, Dynamo, and albumentations but lacks GitPython, which
the benchmark driver needs. Set `VLLM_USE_FLASHINFER_SAMPLER=0` there through
`--model-runtime-env`. Mount a persistent Lustre cache directory at `/cache`
so CUDA, vLLM, and Triton caches survive containers. Mount the dataset and model
cache read-only; mount results writable. Resolve data and model paths through
the YAML `paths` entries.

```bash
python benchmarking/run.py --config benchmarking/gb200-eai-10k.yaml \
  --session-name gb200-eai-10k-20260911 --entries-exact smoke_1pdf_attempt2
python benchmarking/run.py --config benchmarking/gb200-eai-10k.yaml \
  --session-name gb200-eai-10k-20260911 --entries-exact pilot_100pdf_8clients_48sem
```

After validation, launch each full entry on its own four-GPU node:

| Entry | Clients per replica | Semaphore per client | Maximum in-flight requests |
|---|---:|---:|---:|
| `eai_10k_8clients_48sem` | 8 | 48 | 1536 |
| `eai_10k_12clients_48sem` | 12 | 48 | 2304 |
| `eai_10k_8clients_64sem` | 8 | 64 | 2048 |
| `eai_10k_12clients_64sem` | 12 | 64 | 3072 |

These are upper bounds, not measured concurrency. `--inference-batch-size`
limits concurrent HTTP requests per client; it does not set the server's
vLLM batch size. Keep engine defaults, 25 PDFs/task, 300 DPI, 645 pages/PDF,
and 9000 output tokens fixed across the sweep. Full runs must complete 10,000
PDFs and 145,327 pages to pass the entry requirements. Allow four hours per
job; the runner timeout is 12,600 seconds, leaving 30 minutes for setup and
cleanup. Do not reuse an existing entry directory for a retry.

Rank successful full runs using `throughput_pages_per_sec`, along with output
token throughput and quality counters in `tasks.pkl`. Do not rank the smoke
or pilot using `inference_stage_pages_per_sec_per_gpu`: dividing stage time by
configured parallelism inflates that estimate when there are too few tasks.

The runner archives logs under
`ENTRY/ray_cluster/session_latest/nemo_curator_dynamo_*/`. Check
`Dynamo_Frontend.log` for HTTP failures/latency and `Dynamo_DP*.log` for running
and waiting requests, KV-cache usage, and generation throughput. Correlate
these with `ENTRY/gpustats.csv`; low queueing alone does not prove GPU
saturation. Warm-up and drain periods should be distinguished from steady
state. A single run per configuration identifies candidates, not a statistically
established optimum.
