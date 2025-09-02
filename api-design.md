# NeMo Curator Ray API Design

**Status:** Pre Release

## Table of Contents

1. [Background](#background)
2. [Design Principles](#design-principles)
3. [Core Components](#core-components)
   - [Tasks](#tasks)
   - [Stages](#stages)
   - [Pipelines](#pipelines)
   - [Executors](#executors)
4. [Examples](#examples)

## Background

### Current State
- **Existing NeMo-Curator OSS:** Built on Dask for Text, Image, Synthetic Data Generation and Hard negative mining
- **Cosmos Curate:** Built on Ray core with [Cosmos Xenna](https://github.com/nvidia-cosmos/cosmos-xenna) for streaming map-style pipeline
- **API Pattern:** Stages/modules accept distributed Dask dataframes as input

### Why Ray?
- **Unified backend** for all modalities (text, image, video)
- **Better heterogeneous computing** support allowing interleaving of CPU and GPU stages
- **Fractional GPU support** (multiple models per GPU) and multi-GPU support for larger models
- **Enhanced autoscaling** for dynamic workloads

### Why not Ray?
- Limited support for distributed dataframes/arrays, especially distributed operations like groupby
- Eager computation model (addressed through lazy evaluation at curator level)
- Requires Curator to build a physical plan, and implement optimizations like task fusion
## Design Principles

### Task-Centric Architecture
Unlike the previous dataset-level operations, the new design operates on individual **Tasks** - batches of data that flow through the pipeline. This enables:
- Finer-grained control and monitoring
- Better resource utilization

### Map-style (Data-Parallel) Execution
All stages are designed to be map-style on tasks, meaning they take task as input and produce task as output. This allows for easy parallelization and scaling.
- We do not enforce 1-1 mapping between input and output tasks, but rather allow for multiple output tasks from a single input task and multiple input tasks from a single output task. More specifically, a stage applies a transformation from `X` to `Y`, where both `X` and `Y` can be `Task | list[Task] | None`.

### Fault Tolerance Requirements
**All stages MUST be fault-tolerant and retry-safe.** This is a critical requirement because:

- **Task Preemption:** Xenna can preempt/kill running tasks before completion and potentially reschedule them later, especially during autoscaling events
- **Partial Operations:** Tasks may be interrupted mid-execution, leaving partial state (e.g., incomplete file downloads)

## Core Components

1. **Tasks** : The fundamental unit of data that flows through the curation pipeline.
2. **Stages** : The fundamental processing unit that takes Tasks as input and produces Tasks as output.
3. **Pipelines** : A collection of stages that defines the complete processing workflow.
4. **Executors** : The component that orchestrates the execution of the pipeline.

### Tasks

A **Task** is the fundamental unit of data that flows through the curation pipeline, representing a batch of input data for processing.

#### Base Task Implementation

```python
@dataclass
class Task(ABC, Generic[T]):
    """Abstract base class for tasks in the pipeline."""
    task_id: str
    dataset_name: str
    data: T
    _stage_perf: list[StagePerfStats] = field(default_factory=list)
    _metadata: dict[str, Any] = field(default_factory=dict)

    @property
    @abstractmethod
    def num_items(self) -> int:
        """Get the number of items in this task."""

    @abstractmethod
    def validate(self) -> bool:
        """Validate the task data."""
```

#### Example Task Types

#### DocumentTask
For text-based curation pipelines:

```python
@dataclass
class DocumentBatch(Task[pa.Table | pd.DataFrame]):
    """Task for document processing."""

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return isinstance(self.data, pd.DataFrame) and not self.data.empty
```


### Stages


#### Base Stage Interface

```python
class ProcessingStage(ABC, Generic[X, Y], metaclass=StageMeta):
    """Base class for all processing stages that accepts a task of type X and outputs a task of type Y."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this stage."""

    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        return Resources(cpus=1.0)

    @abstractmethod
    def process(self, task: X) -> Y | list[Y]:
        """Process a single task, can output one or more task."""
```

#### Resource Specification

```python
@dataclass
class Resources:
    """Define resource requirements for a processing stage."""
    cpus: float = 1.0 # Number of CPU cores
    gpu_memory_gb: float = 0.0 # Number of GPU memory in GB (Only for single GPU)
    gpus: float = 0.0 # Number of GPUs (Only for multi-GPU)
    nvdecs: int = 0 # Number of NVDEC decoders
    nvencs: int = 0 # Number of NVENC encoders
    entire_gpu: bool = False # Whether to use the entire GPU
```

### Pipelines

A **Pipeline** is a collection of stages that defines the complete processing workflow.

```python
class Pipeline:
    """A pipeline defines a sequence of processing stages."""

    def __init__(self, stages: list[ProcessingStage]):
        self.stages = stages

    def add_stage(self, stage: ProcessingStage):
        """Add a stage to the pipeline."""

    def run(self, executor: BaseExecutor | None = None) -> list[Task] | None:
        """Run the pipeline."""
```

### Executors (Advanced)

**Executors** are responsible for running pipelines on different backends while maintaining a unified interface.
They do so with the help of **Adapters** which are the translation piece between our `ProcessingStage` and the desired "executor".
Each Executor runs a `list[ProcessingStage]` and then wraps each `ProcessingStage` to an `Adapter`, and then finally those wrapped classes, i.e adapters are executed.

#### Base Executor Interface

```python
class BaseExecutor(ABC):
    """Executor for a pipeline."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> None:
        """Execute the pipeline."""
```

### Backend Implementations

#### Xenna Executor
```python
class XennaExecutor(BaseExecutor):
    """Ray-based executor using Xenna backend."""

    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> None:
        # Convert stages to Xenna acceptable format using Xenna Adapters
        # Handle resource allocation
        # Execute with autoscaling
```

#### Ray Data Executor
```python
class RayDataExecutor(BaseExecutor):
    """Ray Data-based executor."""

    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> None:
        # Convert to Ray Data operations
        # Execute pipeline
```

## Examples

Please refer to the [quickstart](./nemo_curator/examples/quickstart.py) for a basic example.
