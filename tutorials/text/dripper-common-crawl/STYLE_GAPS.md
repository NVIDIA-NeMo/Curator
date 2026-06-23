# Style Gaps: SemanticDedup Tutorial vs Dripper Tutorial

## Swarm Results (2026-06-14)

### Fixed in 4-agent swarm

**Agent 1 (P1 Critical Bugs)**
- Added `_convert_main_html()` to stage.py (was missing, broke propagation_stage.py)
- Fixed `DripperHTMLExtractionStage._coerce_html` → module-level `_coerce_html()` in stage.py
- Replaced assert statements with explicit RuntimeError in propagation_stage.py
- Added missing `@dataclass(kw_only=True)` to DripperHTMLPreprocessStage
- Fixed test_stage.py import paths (were importing deleted symbols from stage.py)

**Agent 2 (Field Reduction)**
- DripperHTMLLayoutTemplateStage: 61 → 30 fields
- Created DripperLayoutAdvancedConfig for 12 CC-scale tuning knobs
- Fixed 14 output column name overrides (now use _DRIPPER_*_COL constants)

**Agent 3 (Tutorial → Library Migration)**
- LBP static/dynamic split logic moved to propagation_stage.py
- stage3_cpu_propagation.py: 795 → 674 lines
- stage_gpu_pipeline.py: 648 → 541 lines (uses DripperHTMLPostprocessStage)

**Agent 4 (layout_template.py Size)**
- layout_template.py: 1,872 → 1,569 lines (-303 lines)
- Planning functions extracted to module level (_layout_planning.py: 431 lines)
- Exception handling tightened

### New gaps identified (Iteration 7+)

**Gap 7.1 — stage3_ray_propagation.py reimplements 6 helpers already in the library**
- File: `tutorials/text/dripper-common-crawl/stage3_ray_propagation.py` lines 81–210
- `_coerce_html` (line 81), `_parse_mapping_json` (line 104), `_token_f1` (line 135),
  `_load_cluster_manifest_shard` (line 153), `_load_inference_results` (line 183),
  `_atomic_write_parquet` (line 207) are all re-implemented locally.
- The library already exports `_coerce_html`, `_token_f1`, `_atomic_write_parquet`-equivalent
  from `nemo_curator.stages.text.experimental.dripper.stage` and `_url_helpers`.
- The local `_coerce_html` (line 81–84) skips `_strip_xml_incompatible_chars` and
  `_decode_html_bytes` that the library version applies, creating a silent divergence.
- **Fix:** Replace all 6 local copies with imports from the library. The local
  `_coerce_html` divergence is a correctness risk — the library version must be used.
  Estimated removal: ~60 lines.

**Gap 7.2 — stage3_ray_propagation.py uses stdlib `logging` not loguru (1,080 lines)**
- File: `tutorials/text/dripper-common-crawl/stage3_ray_propagation.py` line 44, 58
- `import logging` + `logger = logging.getLogger(__name__)` — not loguru.
- stage3_cpu_propagation.py already uses `from loguru import logger` (line 46).
- The two Stage 3 variants have inconsistent logging: structured loguru in the
  ProcessPoolExecutor variant, stdlib in the Ray variant.
- **Fix:** Replace `import logging` / `logging.getLogger` with `from loguru import logger`
  at line 44/58. This is a one-line swap; loguru is already in the project deps.

**Gap 7.3 — `_make_stage_cls` in stage_gpu_pipeline.py still uses the anonymous factory pattern**
- File: `tutorials/text/dripper-common-crawl/stage_gpu_pipeline.py` lines 122–154
- Despite Agent 3 migrating postprocessing to `DripperHTMLPostprocessStage`, Stage 1c and
  Stage 2 are still wrapped via `_make_stage_cls(stage_name, setup_fn, process_fn)` which
  produces anonymous classes with no stable `name` attribute and no import path.
- The `process_batch` override (line 144–151) reconstructs a `DocumentBatch` without
  preserving `_metadata` or `_stage_perf`, silently dropping pipeline telemetry.
- **Fix:** Replace the Stage 1c anonymous stage with `DripperHTMLPreprocessStage` (already
  in `preprocessing.py`) and the Stage 2 LLM call with `DripperHTMLInferenceStage` from
  `inference.py`. `_make_stage_cls` can then be deleted entirely (~33 lines removed).

**Gap 7.4 — layout_template.py `process()` carries 3 noqa complexity suppressions**
- File: `nemo_curator/stages/text/experimental/dripper/layout_template.py` line 498
- `def process(...)` is decorated `# noqa: C901, PLR0912, PLR0915` (too-complex,
  too-many-branches, too-many-statements).
- The method dispatches plan execution, collects results, writes output columns, and
  handles timing — all in one function body that was only partially split by Agent 4.
- **Fix:** Extract the output-column assembly loop (currently lines ~580–625) into
  `_assemble_output_df(df, row_results) -> pd.DataFrame` and the plan-dispatch loop into
  `_execute_plans_async(ctx, plans) -> dict`. This should remove all three noqa suppressions.

**Gap 7.5 — `stage.py` `_run_dripper_health_check` silently accepts `RuntimeError` re-raise without re-raise guard**
- File: `nemo_curator/stages/text/experimental/dripper/stage.py` lines 219–226
- The health-check catches all non-RuntimeError exceptions and re-raises as `RuntimeError`,
  but the `except RuntimeError: raise` guard (line 219–220) is a bare re-raise that lets
  `RuntimeError` from `client.query_model` propagate with no additional context.
- The empty-response guard (line 226) uses a no-`EM101` string literal directly in
  `raise RuntimeError(...)` without assigning to a variable first — ruff `EM101` is
  suppressed via the `# noqa: EM101` comment rather than fixed.
- **Fix:** Assign the error string to `msg` before raising (matching the pattern used
  elsewhere in the file). Add `f"Dripper LLM health check timed out or returned no data "
  f"(model={model_name!r})"` as the RuntimeError message so the caller sees the model name.

### PR Status
- Total Python LOC: 13,957 (8,755 tutorial + 5,012 library + 190 workflow)
- F1 (5 retests): 0.8442–0.8443 stable
- Ruff: All checks passed

---

## Status Update (2026-06-14)

### Completed ✅
- Priority 1 (quickstart): ✅ 344→145 lines
- Priority 2 (loguru): ✅ 43 print() eliminated
- Priority 3 (DripperConfig): ✅ dataclass + YAML bridge
- Priority 4 (test_workflow): ✅ 10 synthetic tests, 152 lines
- Priority 5 (type annotations): ✅ completed
- Item 6 (WorkflowRunResult): ✅ typed return

---

## Iteration 2-4 Architectural Improvements

- stage.py split: 3,776→489 lines (-87%)
- layout_template.py extracted: 2,356 lines focused file
- stage.py now only 489 lines (shared utilities)
- workflow.py: WorkflowRunResult return type
- quickstart.py: 344→145 lines
- test_workflow.py: new, 152 lines
- 4 consecutive cluster retests: F1=0.8442~0.8443 confirmed stable

---

## Remaining Gaps (Iter 5+)

- layout_template.py still 2,356 lines (SemanticDedup equivalent: ~322)
- stage3_cpu_propagation.py: 902 lines
- run_pipeline.py: 723 lines (Slurm orchestrator, inherently cluster-specific)
- pipeline_metrics.py: 265 lines (could use Curator's built-in metric tracking)

---

**Date:** 2026-06-14
**Scope:** Code style and maintainability comparison between `SemanticDeduplicationWorkflow`
(the established pattern in `nemo_curator/stages/deduplication/semantic/workflow.py` and its
image tutorial `tutorials/image/getting-started/image_dedup_example.py`) versus the Dripper
CC-scale tutorial scripts under `tutorials/text/dripper-common-crawl/`.

---

## 1. Entry Point / User API

**SemanticDedup approach:**
```python
# tutorials/image/getting-started/image_dedup_example.py — 8 lines to run the full pipeline
pipeline = SemanticDeduplicationWorkflow(
    input_path=args.embeddings_dir,
    output_path=args.removal_parquets_dir,
    id_field="image_id",
    embedding_field="embedding",
    n_clusters=100,
    eps=0.01,
)
pipeline.run(pairwise_executor=executor)  # single call; returns WorkflowRunResult
```

**Dripper current approach:**
```bash
# To run the full pipeline the user must:
# 1. Edit configs/template.yaml with cluster paths, model params, resource overrides
# 2. python run_pipeline.py --config configs/template.yaml
#    → SSH to a login node, generate 7+ sbatch scripts, submit them one by one via aftercorr
# 3. Monitor 7 Slurm array jobs (stage1a/1b/gpu_pipeline/stage3/stage3b_build/3b_gpu/3b_merge)
# 4. Optionally call: python compare_f1.py --baseline ... --pipeline ...
```

**Gap:** The Dripper tutorial has no single Python entry point that a developer can call
in a local or CI environment. The "entry point" (`run_pipeline.py`) is a Slurm-SSH
orchestrator that requires a live cluster with hardcoded Lustre paths, not a composable
Python API. A reviewer cannot run `python tutorial.py` to see the pipeline work.

**Fix:** Mirror the `DripperHTMLWorkflow` class (already in
`nemo_curator/stages/text/experimental/dripper/workflow.py`) in the tutorial by adding a
`demo.py` or `quickstart.py` that instantiates `DripperHTMLWorkflow` and calls
`workflow.run(executor)` — the same one-liner pattern the SemanticDedup image tutorial
uses.

---

## 2. Stage Construction Pattern

**SemanticDedup approach:**
```python
# Internally, SemanticDeduplicationWorkflow builds stages in _run_kmeans_stage /
# _run_pairwise_stage via named, typed constructors:
kmeans_stage = KMeansStage(
    n_clusters=self.n_clusters,
    id_field=self.id_field,
    embedding_field=self.embedding_field,
    input_path=self.input_path,
    output_path=self.kmeans_output_path,
    ...
)
pipeline.add_stage(kmeans_stage)
```

**Dripper current approach:**
```python
# stage_gpu_pipeline.py — stages are constructed dynamically via a factory function
# that builds anonymous ProcessingStage subclasses closed over free callables:
def _make_stage_cls(stage_name: str, setup_fn: Callable, process_fn: Callable) -> type:
    """Build a NeMo ProcessingStage class, cached by stage_name."""
    class _Stage(ProcessingStage[_DocumentBatch, _DocumentBatch]):
        name = stage_name
        resources = Resources(cpus=1.0)
        batch_size = 1
        def setup(self, _worker_metadata=None): setup_fn()
        def process_batch(self, tasks): ...
    _STAGE_CLS_CACHE[stage_name] = _Stage
    return _Stage
```

**Gap:** The dynamic `_make_stage_cls` pattern produces anonymous, unconfigurable stage
classes that are harder to introspect, test, and reuse. There is no stable class name to
`isinstance`-check or import in tests. The SemanticDedup pattern uses named, first-class
`ProcessingStage` subclasses (`KMeansStage`, `PairwiseStage`) that can be imported and
composed independently.

**Fix:** Replace `_make_stage_cls` with proper named `ProcessingStage` subclasses
(e.g. `DripperHTML1cPreprocessStage`) that live in `nemo_curator/stages/`. The workflow
file already does this correctly for the library-level stages; the tutorial should import
them rather than reinvent them.

---

## 3. Configuration

**SemanticDedup approach:**
```python
# All configuration is expressed as typed __init__ parameters with defaults:
# nemo_curator/stages/deduplication/semantic/workflow.py
class SemanticDeduplicationWorkflow(WorkflowBase):
    def __init__(
        self,
        input_path: str | list[str],
        output_path: str,
        n_clusters: int,
        eps: float | None = None,
        distance_metric: Literal["cosine", "l2"] = "cosine",
        which_to_keep: Literal["hard", "easy", "random"] = "hard",
        verbose: bool = True,
        ...
    ):
```

**Dripper current approach:**
```yaml
# configs/template.yaml — resource and model params in YAML
resources:
  gpu_pipeline:
    model: "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"
    max_tokens: 2048
    gpu_mem_util: 0.90
    max_num_seqs: 512
```
```python
# stage_gpu_pipeline.py — same params duplicated as argparse arguments:
p.add_argument("--model", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
p.add_argument("--max-tokens", type=int, default=2048)
p.add_argument("--gpu-mem-util", type=float, default=0.90)
p.add_argument("--max-num-seqs", type=int, default=512)
```

**Gap:** Model and resource parameters are defined twice: once in `configs/template.yaml`
and once as `argparse` defaults in each stage script. There is no single authoritative
source of truth. Adding a new parameter requires editing both files; defaults can silently
diverge. The YAML schema is also undocumented (no schema validation or dataclass mapping).

**Fix:** Map the YAML config directly onto the `DripperHTMLWorkflow` dataclass fields.
Provide a `DripperConfig.from_yaml(path)` classmethod that validates types, so the YAML
becomes a serialization of the typed Python config rather than a separate parallel format.

---

## 4. LOC Comparison

| File | LOC | Purpose |
|---|---|---|
| `image_dedup_example.py` (SemanticDedup tutorial) | 301 | Full runnable image dedup pipeline |
| `nemo_curator/stages/deduplication/semantic/workflow.py` | 431 | Library workflow class |
| **SemanticDedup total** | **732** | |
| `stage_gpu_pipeline.py` | 660 | Combined stages 1c+2+2b |
| `stage3_cpu_propagation.py` | 858 | Stage 3 propagation |
| `run_pipeline.py` | 718 | Slurm orchestrator |
| `compare_f1.py` | 143 | Validation script |
| `stage1b_gpu_dbscan.py` | 357 | Stage 1b clustering |
| `stage1c_cpu_preprocess.py` | 137 | Stage 1c preprocessing |
| `stage3b_fallback_llm.py` | 135 | Stage 3b fallback |
| `pipeline_metrics.py` | 265 | Metrics tracking |
| **Dripper tutorial total** | **3,273** | (tutorial scripts only) |
| **Total dripper lines added in PR** | **~9,114** | (git diff stat) |

**Gap:** The Dripper tutorial is 4.5x larger than the SemanticDedup tutorial to express a
conceptually similar "run pipeline, get output" operation. Much of this LOC lives in
bespoke SSH/Slurm orchestration, inline subprocess management, and duplicated argparse
boilerplate that the SemanticDedup pattern encapsulates in reusable library classes.

**Fix:** Move the reusable logic (stage classes, argparse defaults, metrics) into the
library (`nemo_curator/stages/text/experimental/dripper/`). The tutorial should thin down
to ~150–200 LOC, importing from the library rather than reimplementing it.

---

## 5. Error Handling

**SemanticDedup approach:**
```python
# nemo_curator/stages/deduplication/semantic/workflow.py
def run(self, ...):
    try:
        self._setup_directories()
        ...
        return workflow_result
    except Exception as e:
        logger.error(f"Semantic deduplication pipeline failed: {e}")
        raise  # re-raise so the caller sees the original exception and traceback
```
Configuration errors are caught eagerly in `_validate_config()` with typed `ValueError` /
`TypeError` before any compute begins.

**Dripper current approach:**
```python
# stage_gpu_pipeline.py — bare except swallows errors into the output record
try:
    case = _b.case_cls(_b.input_cls(raw_html=html, url=url))
    ...
except Exception as exc:
    out["prompt"] = f"ERROR:{type(exc).__name__}:{str(exc)[:100]}"

# stage3_cpu_propagation.py — similar pattern
try:
    ...
except Exception as exc:
    logger.debug("loader failed; trying next")

# stage3_cpu_propagation.py — corrupt-file recovery silently unlinks
try:
    meta = pq.read_metadata(str(out_path))
except OSError:
    out_path.unlink(missing_ok=True)  # corrupt file — remove and reprocess
```

**Gap:** Dripper tutorials use broad `except Exception` guards in many hot-path functions,
converting errors into silent per-record error strings or log-only debug messages. This
means a systematic misconfiguration (wrong model path, missing column) can process
millions of pages and only be detected by inspecting `dripper_error` fields in output
parquet files rather than raising at startup. The SemanticDedup pattern validates eagerly
and re-raises so CI detects failures immediately.

**Fix:** Add a `validate()` method (or call it from `DripperHTMLWorkflow.__post_init__`)
that checks required inputs before any Ray workers are spawned. Reserve broad per-record
exception capture only for the innermost HTML-parsing call, and surface aggregate error
counts via the `WorkflowRunResult` metadata rather than silent sentinel strings.

---

## 6. Type Annotation Completeness

**SemanticDedup approach:**
```
nemo_curator/stages/deduplication/semantic/workflow.py: 5/7 functions annotated (71%)
nemo_curator/stages/text/experimental/dripper/workflow.py: 2/2 functions annotated (100%)
```
All public methods have full return-type annotations. `__init__` parameters use
`str | list[str]`, `Literal[...]`, typed defaults throughout.

**Dripper current approach:**
```
tutorials/text/dripper-common-crawl/stage_gpu_pipeline.py:  19/21 annotated (90%)
tutorials/text/dripper-common-crawl/stage3_cpu_propagation.py: 20/31 annotated (65%)
tutorials/text/dripper-common-crawl/compare_f1.py: 5/5 annotated (100%)
```
Notable unannotated functions in `stage3_cpu_propagation.py`:

```python
# Missing return type on several private helpers (31 total, 11 unannotated):
def _apply_ratio_guard(content, url, prop_config):  # no -> annotation
def _try_lbp_once(row, prop_config):                # no -> annotation
def _sibling_propagate(siblings, gpu_row, ...):     # no -> annotation
def _make_rep_or_singleton_row(row, role):          # no -> annotation
def _make_fallback_row(row, role, error):           # no -> annotation
```

**Gap:** `stage3_cpu_propagation.py` has 65% annotation coverage — a 35-point gap from
the SemanticDedup library style. Missing annotations on functions with complex return
types (`dict[str, Any]`, `list[dict]`) make it harder for mypy and IDE tooling to catch
bugs at authorship time.

**Fix:** Add `-> dict[str, Any]` / `-> list[dict[str, Any]]` / `-> None` to the 11
unannotated public and private helpers in `stage3_cpu_propagation.py`. Enable `mypy` in
CI for the tutorial directory with `--ignore-missing-imports`.

---

## 7. Logging Style

**SemanticDedup approach:**
```python
# nemo_curator/stages/deduplication/semantic/workflow.py
from loguru import logger   # single consistent import

logger.info("Starting K-means clustering stage (RayActorPoolExecutor)...")
logger.success(f"K-means clustering completed in {kmeans_time:.2f} seconds")
logger.warning(
    f"n_clusters={self.n_clusters} is less than {MIN_RECOMMENDED_N_CLUSTERS}. ..."
)
logger.error(f"Semantic deduplication pipeline failed: {e}")
# 38 logger.* calls; 0 print() calls in the workflow
```

**Dripper current approach (mixed, inconsistent):**
```python
# stage_gpu_pipeline.py — uses print() with flush=True, no logger at all
print(f"[gpu-pipeline] Stage 1c: {ok:,}/{len(df):,} prompts in {elapsed:.1f}s", flush=True)
print(f"[gpu-pipeline] Stage 2: {len(df):,} pages over {n_gpus} GPUs", flush=True)
print(f"[gpu-pipeline] ALL DONE: ...", flush=True)
# 0 logger.* calls

# stage3_cpu_propagation.py — uses stdlib logging.getLogger AND print() in the same file
logger = logging.getLogger(__name__)       # stdlib, not loguru
...
logger.debug("pickle.loads from bytes failed; trying string decode")
print(f"[stage3] shard {shard_index}: {len(tasks):,} cluster tasks...", flush=True)
# 2 logger.* calls, 12 print() calls

# compare_f1.py — print() only, 19 calls
print("[f1] loading baseline...", flush=True)

# run_pipeline.py — logging.getLogger AND 5 print() calls
logger = logging.getLogger(__name__)
```

**Gap:** Across the four main Dripper tutorial files there are 43 `print()` calls and
only 7 `logger.*` calls (all using stdlib `logging`, not `loguru`). The `[stage-prefix]`
convention embedded in print strings is a manual workaround for the structured context
loguru provides natively. This makes it impossible to globally adjust log levels, redirect
to files, or suppress output in tests without patching `sys.stdout`.

**Fix:** Replace all `print(f"[stage3] ...", flush=True)` calls with
`logger.info("...")` using `loguru` (matching the library convention). In test code, use
`loguru`'s `caplog`/`capfd` sink rather than patching stdout.

---

## 8. Test Coverage Style

**SemanticDedup approach:**
```python
# tests/stages/deduplication/semantic/test_workflow.py
class TestSemanticDeduplicationWorkflow:
    def setup_method(self):
        # Creates synthetic blobs in memory; no Slurm, no cluster needed
        self.X, _ = make_blobs(n_samples=..., n_features=3, random_state=42)
        self.df = pd.DataFrame({"id": ..., "embeddings": self.X.tolist()})

    def test_semantic_deduplication_with_duplicate_identification(self, tmpdir, ...):
        pipeline = SemanticDeduplicationWorkflow(
            input_path=input_dir, output_path=output_dir,
            n_clusters=self.n_clusters, eps=0.01, ...
        )
        results = pipeline.run(pairwise_executor=executor)
        assert results.get_metadata("total_time") > 0
        assert duplicates_identified == expected_removed   # exact count verified
```
Tests exercise the full Python API end-to-end; no subprocess spawning, no SSH, no Slurm.

**Dripper current approach:**
```python
# tests/stages/text/experimental/dripper/test_stage.py
# Tests the underlying stage classes (good), but tests the tutorial-level
# orchestration only via the test_pipeline_correctness.py which:
# - Requires a running Ray cluster
# - Reads from filesystem paths set via environment variables
# - Has no synthetic data generation (needs pre-existing parquet files)
# tutorials/text/dripper-common-crawl/test_pipeline_correctness.py:
#   "Run full pipeline on a small subset and verify F1 > threshold"
#   → this is an integration test masquerading as a unit test
```

**Gap:** The Dripper library-level stage tests are good (`test_stage.py`), but the
tutorial has no self-contained unit test for the orchestration layer (the equivalent of
`test_workflow.py` for SemanticDedup). The only end-to-end test requires a live cluster.
SemanticDedup's test synthesizes data in-process and verifies exact duplicate counts,
giving immediate CI feedback.

**Fix:** Add a `tests/stages/text/experimental/dripper/test_workflow.py` that instantiates
`DripperHTMLWorkflow` with a `FakeAsyncLLMClient`, generates a tiny in-memory HTML
dataset, runs the pipeline via `XennaExecutor`, and asserts on output column presence and
content length > 0. Mirror the `setup_method` / `tmpdir` pattern from
`test_workflow.py`.

---

## 9. Documentation and Docstrings

**SemanticDedup approach:**
```python
# nemo_curator/stages/deduplication/semantic/workflow.py — class-level docstring:
class SemanticDeduplicationWorkflow(WorkflowBase):
    """
    End-to-End Semantic Deduplication Workflow.
    It consists of the following stages:
    - KMeansStage: ...
    - PairwiseStage: ...
    - IdentifyDuplicatesStage (optional): ...
    """

    def __init__(self, ...):
        """
        Initialize the semantic deduplication workflow.

        Args:
            input_path: Directory or list of directories containing input files with embeddings
            output_path: Directory to write output files (i.e. ids to remove)
            n_clusters: Number of clusters for K-means
            eps: Epsilon value for duplicate identification
            ...  # every parameter documented
        """
```

**Dripper current approach:**
```python
# stage_gpu_pipeline.py — module docstring only, no class or __init__ docstrings
"""Combined Stage 1c + Stage 2 + Stage 2b in a single GPU job.

Eliminates two intermediate parquet round-trips and two Slurm queue waits.
INPUT:  Stage 1b output dir. OUTPUT: combined parquet with Stage 2b schema.
RUNS ON: batch GPU partition (8xH100). Replaces JOB1c + JOB2 + JOB2b.
"""
# _WorkerConfig dataclass has no field-level docstring:
@dataclass
class _WorkerConfig:
    model: str
    gpu_mem_util: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_tokens: int
    kv_cache_dtype: str
    # No description of what each field does

# DripperHTMLWorkflow (in nemo_curator/stages/text/experimental/dripper/workflow.py)
# has good class + field docstrings — but the tutorial files that call it do not.
```

**Gap:** The tutorial stage scripts (`stage_gpu_pipeline.py`, `stage3_cpu_propagation.py`)
have module-level docstrings and per-function docstrings on most private helpers, but no
`Args:` / `Returns:` sections in the Google/NumPy style used by the SemanticDedup
workflow. The `_WorkerConfig` and `_HyperParams` dataclasses lack field-level
documentation. A newcomer cannot tell which fields are required vs. optional or what the
units are (e.g. `gpu_mem_util` is a fraction 0.0–1.0, not a percentage).

**Fix:** Add `Args:` / `Returns:` sections to the 10 public-facing functions in the
tutorial scripts. Add field comments (`#: fraction of GPU memory, 0.0–1.0`) to
`_WorkerConfig` and `_HyperParams`.

---

## 10. Overall LOC in PR vs SemanticDedup Baseline

```bash
# git diff origin/main --stat | grep -E "dripper|tutorial" | tail -5
 tutorials/text/dripper-common-crawl/stage3b_fallback_llm.py |  135 +
 tutorials/text/dripper-common-crawl/stage_gpu_pipeline.py   |  660 ++++
 tutorials/text/dripper-common-crawl/run_pipeline.py         |  718 ++++
 tutorials/text/dripper-common-crawl/stage3_cpu_propagation.py | 858 +++++
 Total lines added (dripper + tutorial):                      ~9,114
```

Compared to SemanticDedup (library + tutorial) which totals **732 lines** for full
end-to-end coverage, the Dripper PR adds **12.4x** more code to express a pipeline that
could theoretically be expressed in the same idiom. A large fraction of this overhead is:

- Slurm/SSH orchestration that belongs in a cluster-specific runner, not the tutorial
- Bespoke argparse blocks repeated across 6 stage scripts (instead of one config dataclass)
- Inline `sys.path` manipulation (`sys.path.insert(0, str(Path(__file__).parent))`)
- `print(flush=True)` plumbing repeated instead of a shared logger

---

## Prioritized TODO List

### Priority 1 — Add a self-contained quickstart entry point
**Impact: Discoverability, testability**
Create `tutorials/text/dripper-common-crawl/quickstart.py` (~100 LOC) that:
- Instantiates `DripperHTMLWorkflow` from the library
- Uses a `FakeAsyncLLMClient` or a local model for smoke-test
- Calls `workflow.run(XennaExecutor())`
- Prints a summary table of results
This eliminates the "must have a Slurm cluster to try Dripper" barrier for new
contributors.

### Priority 2 — Unify logging to loguru
**Impact: Debuggability, test isolation**
Replace all 43 `print(f"[stage-prefix] ...", flush=True)` calls in the four main tutorial
files with `from loguru import logger; logger.info(...)`. Remove `logging.getLogger`
usage in tutorial files (keep it only where stdlib `logging` is truly required for a
third-party library). This makes it possible to suppress output in tests and redirect to
files in production with a one-line sink configuration.

### Priority 3 — Eliminate YAML/argparse configuration duplication
**Impact: Maintainability, correctness**
Add a `DripperConfig` dataclass (or extend `DripperHTMLWorkflow` fields) that can be
serialized to/from YAML. Remove the parallel argparse defaults in each stage script that
duplicate `configs/template.yaml`. A single `DripperConfig.from_yaml(path)` classmethod
provides one authoritative source of truth for all parameters.

### Priority 4 — Add a `test_workflow.py` with synthetic data
**Impact: CI coverage, regression prevention**
Mirror `tests/stages/deduplication/semantic/test_workflow.py` for Dripper: a
`TestDripperHTMLWorkflow` class that builds a 10-row HTML dataset in memory, runs the
full pipeline with a fake client, and asserts on output columns and non-empty content.
This gives the same level of API coverage that SemanticDedup has without requiring a
Slurm cluster.

### Priority 5 — Complete type annotations in `stage3_cpu_propagation.py`
**Impact: Type safety, IDE support**
Add return-type annotations to the 11 unannotated functions
(`_apply_ratio_guard`, `_try_lbp_once`, `_sibling_propagate`,
`_make_rep_or_singleton_row`, `_make_fallback_row`, and 6 others). Add
field-level docstrings to `_WorkerConfig` and `_HyperParams`. Enable `mypy` in CI for
the tutorial directory. This closes the 35-point annotation gap relative to the
SemanticDedup library style and will catch the next `dict` vs `list` confusion at
type-check time rather than at runtime.

---

## 6. Return Type from workflow.run()

**SemanticDedup approach:**
```python
result = workflow.run(executor)
result.get_metadata("final_output_path")  # WorkflowRunResult with typed methods
```

**Dripper current approach:**
```python
result = workflow.run(executor)
result["output_tasks"]  # plain dict — no typed access, no metadata protocol
```

**Gap:** DripperHTMLWorkflow.run() returns a plain dict instead of WorkflowRunResult.

**Fix:** Return `WorkflowRunResult` from `nemo_curator.pipeline.workflow`.
