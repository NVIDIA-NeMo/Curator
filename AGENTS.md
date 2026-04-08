# NeMo Curator — Agent & AI Contributor Guide

This file provides guidance for AI coding agents (Copilot, Claude Code, Cursor, etc.) and their operators contributing to NeMo Curator. Read it before writing any code or opening a PR.

---

## Quick Reference

```bash
uv sync --extra all          # install all dependencies
pre-commit run --all-files   # lint + format
pytest -m "not gpu"          # run CPU tests
pytest                       # run all tests (requires GPU)
```

---

## Environment Setup

**Always use `uv`. Never use bare `pip install` or `python -m pip`.**

```bash
# Install uv (one-time)
pip install uv

# Install project + all dev dependencies
uv sync --extra all

# Install just a specific modality
uv sync --extra text_cuda12
uv sync --extra audio_cpu --extra video_cpu
```

Available extras: `text_cpu`, `text_cuda12`, `audio_cpu`, `audio_cuda12`, `image_cpu`, `image_cuda12`, `video_cpu`, `video_cuda12`, `deduplication_cuda12`, `sdg_cpu`, `sdg_cuda12`, `interleaved_cpu`, `interleaved_cuda12`, `all`.

**Set up pre-commit hooks (required before committing):**

```bash
uv sync --extra all          # pre-commit is included via the dev dependency group
uv run pre-commit install --install-hooks
```

---

## Running Tests

```bash
# CPU-only (no GPU required)
pytest -m "not gpu"

# Full suite (requires NVIDIA GPU, Volta or newer, CUDA 12.x)
pytest

# Specific folder
pytest tests/stages/text/
pytest tests/stages/audio/

# With coverage (what CI runs)
coverage run --branch --source=nemo_curator -m pytest -m "not gpu"
coverage report
```

**Test layout mirrors source layout.** New stage in `nemo_curator/stages/text/filters/` → tests in `tests/stages/text/filters/`. GPU-only tests are marked `@pytest.mark.gpu`. CPU tests must pass without any GPU.

**Coverage requirement: 80% on changed lines.** CI will block PRs that don't meet this.

---

## Linting and Formatting

NeMo Curator uses `ruff` for both linting and formatting (line length: 119). The pre-commit hooks run it automatically on commit.

```bash
# Run all checks manually
pre-commit run --all-files

# Auto-fix what ruff can fix
ruff check --fix .
ruff format .
```

Notable ruff config (from `pyproject.toml`):
- Docstrings are **not enforced by ruff** — the entire `D` (pydocstyle) rule group is in the ignore list. However, CONTRIBUTING.md asks contributors to include docstrings for every public class and method as a code-quality convention; ruff just won't fail the CI for missing them.
- Use `loguru.logger`, not `print()`.
- Full type annotations required on all functions.
- `tests/` has relaxed rules (asserts, magic values allowed).

---

## Architecture Overview

NeMo Curator is built around four core abstractions. Understand these before writing any stage or pipeline code.

### 1. Task (`nemo_curator/tasks/`)

A **Task** is the unit of data flowing through a pipeline — a typed batch of data for a single processing step.

```python
@dataclass
class Task(ABC, Generic[T]):
    task_id: str
    dataset_name: str
    data: T                             # payload (DataFrame, Table, etc.)
    _stage_perf: list[StagePerfStats]   # auto-populated by executor
    _metadata: dict[str, Any]           # arbitrary key-value metadata

    @property
    @abstractmethod
    def num_items(self) -> int: ...

    @abstractmethod
    def validate(self) -> bool: ...
```

Pre-built task types: `DocumentBatch` (text), `AudioTask`, `ImageBatch`, `VideoBatch`, `InterleavedBatch`, `FileGroupTask`.

### 2. ProcessingStage (`nemo_curator/stages/base.py`)

A **ProcessingStage** transforms one task type to another. This is where all curation logic lives.

```python
class ProcessingStage(ABC, Generic[X, Y], metaclass=StageMeta):
    # ── Class attributes ── set these, never override as @property ──
    name      = "MyStage"          # unique string identifier (required)
    resources = Resources(cpus=1.0) # declare CPU/GPU needs
    batch_size = 1                  # tasks per batch
    runtime_env: ClassVar[dict | None] = None  # Ray runtime env

    @abstractmethod
    def process(self, task: X) -> Y | list[Y] | None:
        """1-to-1, 1-to-many, or filter (return None to drop task)."""

    # Optional overrides:
    def process_batch(self, tasks: list[X]) -> list[Y]: ...   # vectorized processing
    def inputs(self)  -> tuple[list[str], list[str]]: ...     # declare required attrs/columns
    def outputs(self) -> tuple[list[str], list[str]]: ...     # declare produced attrs/columns
    def setup(self, worker_metadata) -> None: ...             # called once per worker
    def setup_on_node(self, node_info, worker_metadata) -> None: ...  # called once per node
    def teardown(self) -> None: ...                           # cleanup after processing
```

**Critical rules for stages:**
- Set `name`, `resources`, `batch_size`, `runtime_env` as **plain class attributes**, never as `@property`.
- Never override `_name`, `_resources`, or `_batch_size` — these are `@final` properties on the base class and will raise `TypeError`.
- Concrete (non-abstract) stages are **automatically registered** in `_STAGE_REGISTRY` by `StageMeta`. No manual registration needed.
- All stages **must be idempotent and retry-safe**. Executors (Ray, Xenna) can preempt and reschedule tasks mid-execution.

**Customizing a stage without subclassing** — use `with_()`:

```python
stage = MyStage().with_(
    resources=Resources(gpu_memory_gb=40.0),
    batch_size=32,
)
```

### 3. CompositeStage (`nemo_curator/stages/base.py`)

A **CompositeStage** is a user-facing stage that decomposes into multiple execution stages at pipeline-build time. It is never executed directly. Use it when a logical operation requires multiple internal stages (e.g., identify + remove for deduplication).

```python
class MyCompositeStage(CompositeStage[DocumentBatch, DocumentBatch]):
    name = "MyCompositeStage"

    def decompose(self) -> list[ProcessingStage]:
        return [IdentifyStage(), RemoveStage()]
```

### 4. Pipeline (`nemo_curator/pipeline/pipeline.py`)

```python
pipeline = Pipeline("name", "description")
pipeline.add_stage(Stage1()).add_stage(Stage2()).add_stage(Stage3())
pipeline.build()   # decomposes CompositeStages into execution stages
```

### 5. Executors (`nemo_curator/backends/`)

Executors run pipelines on a backend. Two backends exist:
- `RayDataExecutor` — Ray Data streaming pipeline (text/image/audio/video)
- `XennaExecutor` — Cosmos Xenna backend for production deployments

See `tutorials/quickstart.py` for a complete working example.

---

## Adding a New Stage (Step-by-Step)

1. **Pick the right location**: `nemo_curator/stages/{modality}/{category}/your_stage.py`
   - e.g., a new text filter → `nemo_curator/stages/text/filters/my_filter.py`

2. **Write the stage**:

```python
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.document import DocumentBatch


class MyFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """One-line summary for docs.

    Longer description if needed.
    """

    name = "MyFilter"
    resources = Resources(cpus=1.0)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["text"]   # requires task.data to have a "text" column

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["text"]

    def process(self, task: DocumentBatch) -> DocumentBatch | None:
        filtered = task.data[task.data["text"].str.len() > 10]
        if filtered.empty:
            return None   # drop the task
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=filtered,
        )
```

3. **Export it** — add to the appropriate `__init__.py` so users can import it.

4. **Write tests** — mirror the source path: `tests/stages/text/filters/test_my_filter.py`.
   - Test CPU path without any GPU dependency.
   - Mark GPU-only tests with `@pytest.mark.gpu`.
   - Aim for ≥80% coverage of your new code.

5. **Run checks before committing**:

```bash
pre-commit run --all-files
pytest tests/stages/text/filters/test_my_filter.py
```

---

## Declaring GPU Resources

Declare GPU needs in `resources` so the executor schedules correctly:

```python
# Single GPU stage
resources = Resources(gpu_memory_gb=40.0)

# Multi-GPU stage
resources = Resources(gpus=4)

# CPU only (default)
resources = Resources(cpus=1.0)
```

---

## Logging

Use `loguru.logger`. Never use `print()`.

```python
from loguru import logger

logger.info("Processing task {}", task.task_id)
logger.warning("Dropping empty batch for {}", task.dataset_name)
logger.debug("Scores: {}", scores)
```

---

## Commit Requirements

**All commits must be signed (DCO) and signed-off:**

```bash
git commit -sS -m "feat: add MyFilter stage for text quality filtering"
#           ^^
#           -s  = --signoff  (adds Signed-off-by trailer, required by DCO)
#           -S  = GPG sign   (required if you have commit signing configured)
```

If you forget the `-s`:

```bash
git reset --soft HEAD~1
git add <files>
git commit -sS -m "your message"
```

**Commits without `Signed-off-by` will be rejected by CI.**

---

## Updating Dependencies

When you change `pyproject.toml`, regenerate the lock file:

```bash
uv lock
git add pyproject.toml uv.lock
git commit -s -m "chore: update dependencies"
```

The `uv-lock` pre-commit hook will auto-regenerate and block the commit if the lock file is stale — stage the generated file and commit again.

---

## Pull Request Requirements

1. **One PR does one thing.** "What does this PR do?" must have a clear one-sentence answer.
2. **All CI checks must pass** — linting, CPU tests, 80% coverage on changed lines.
3. **Target `main`.**
4. **Disclose AI assistance.** If an AI agent wrote or materially assisted with the code, include in the PR description:

   > *This PR was created with AI assistance (Claude Code / GitHub Copilot / etc.).*

   And add a co-authorship trailer to the commit:

   ```
   Co-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>
   ```

5. **Do not open a PR for trivial formatting-only changes** (single typos, whitespace) unless they are part of a larger change.

---

## What NOT to Do

| Don't | Do instead |
|-------|-----------|
| `pip install ...` | `uv sync --extra <group>` |
| `python script.py` | `uv run python script.py` |
| Define `name` as `@property` on a stage | Use a plain class attribute: `name = "MyStage"` |
| Override `_name`, `_resources`, or `_batch_size` | Use `name`, `resources`, `batch_size` class attributes |
| Call `process()` directly on a `CompositeStage` | Let the Pipeline's `build()` decompose it |
| Use `print()` for logging | Use `loguru.logger` |
| Commit without `-s` (signoff) | `git commit -sS ...` |
| Open a PR without tests | Write tests in `tests/` mirroring the source path |
| Add heavy imports at module level for optional deps | Use lazy imports inside functions |

---

## Key File Locations

| What | Where |
|------|-------|
| Stage base class | `nemo_curator/stages/base.py` |
| Resources dataclass | `nemo_curator/stages/resources.py` |
| Task base class | `nemo_curator/tasks/tasks.py` |
| DocumentBatch (text) | `nemo_curator/tasks/document.py` |
| Pipeline | `nemo_curator/pipeline/pipeline.py` |
| RayData executor | `nemo_curator/backends/ray_data/executor.py` |
| Xenna executor | `nemo_curator/backends/xenna/` |
| Text stages | `nemo_curator/stages/text/` |
| Audio stages | `nemo_curator/stages/audio/` |
| Image stages | `nemo_curator/stages/image/` |
| Video stages | `nemo_curator/stages/video/` |
| Quickstart tutorial | `tutorials/quickstart.py` |
| Ruff config | `pyproject.toml` → `[tool.ruff]` |
| Pre-commit config | `.pre-commit-config.yaml` |
| CI pipeline | `.github/workflows/cicd-main.yml` |
| API design doc | `api-design.md` |
