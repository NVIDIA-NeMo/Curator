# GitHub Copilot Instructions for NVIDIA NeMo Curator

## Overview

NVIDIA NeMo Curator is a scalable data preprocessing tool for training large language models. This repository focuses on data curation pipelines for text, audio, video, and image modalities with support for both CPU and GPU processing.

## Development Environment

### Python Environment
- **Python Version**: 3.12 (recommended and tested), supports 3.10-3.12
- **Package Manager**: [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management  
- **Virtual Environment**: Use uv's built-in virtual environment management
- **Note**: Python 3.11 is not supported due to RAPIDS compatibility requirements

### CUDA Environment (Optional)
- **CUDA Version**: 12.x (12.0+) for GPU acceleration
- **GPU Requirements**: NVIDIA GPU with Volta™ architecture or higher (compute capability 7.0+)
- **GPU Libraries**: RAPIDS (cuDF, cuML), PyTorch with CUDA 12 support, CuPy
- **Note**: GPU features are optional and CPU fallbacks are provided

## Key Technologies and Frameworks

### Core Dependencies
- **PyTorch**: Deep learning framework with CUDA support
- **Ray**: Distributed computing framework for data processing
- **Pandas/CuDF**: Data manipulation (CPU/GPU respectively)
- **Transformers**: Hugging Face transformers library
- **LoguRu**: Structured logging

### Modality-Specific Libraries
- **Text Processing**: PyTorch, BeautifulSoup, fasttext, sentencepiece, trafilatura
- **Audio Processing**: NeMo Toolkit ASR components
- **Video Processing**: OpenCV, PyAV, CvCuda, PyNvVideoCodec
- **Image Processing**: NVIDIA DALI for optimized data loading

### Testing Framework
- **pytest**: Primary testing framework
- **pytest-asyncio**: Async testing support
- **pytest-coverage**: Code coverage measurement
- **GPU Testing**: Use `@pytest.mark.gpu` for GPU-dependent tests

## Setup Instructions

### Basic Setup
```bash
# Install uv package manager
pip3 install uv

# Clone and setup development environment
git clone <repository-url>
cd Curator
uv sync

# For GPU development (requires CUDA 12.x)
uv sync --extra deduplication_cuda12x
```

### Optional Feature Groups
```bash
# Text processing capabilities
uv sync --extra text

# Video processing with GPU acceleration
uv sync --extra video --extra video_cuda

# All features (includes CUDA dependencies)
uv sync --extra all
```

## Coding Standards and Patterns

### Code Quality
- **Linting**: Ruff with comprehensive rule set (see pyproject.toml)
- **Line Length**: 119 characters maximum
- **Type Hints**: Use comprehensive type annotations
- **Imports**: Follow import sorting conventions

### Module Structure
- **Stages**: Processing stages organized by modality (`text/`, `video/`, `audio/`, `image/`)
- **Utils**: Shared utilities for common operations
- **Datasets**: Data loading and manipulation classes
- **Modules**: Core processing algorithms

### Error Handling Patterns
```python
# Standard error handling for missing dependencies
try:
    import required_library
except ImportError as e:
    logger.error(f"Required dependency not found: {e}")
    raise ImportError("Please install the required dependencies")
```

### Configuration Patterns
- Use YAML configuration files for processing pipelines
- Support hierarchical configuration (CLI args > env vars > config files > defaults)
- Follow the configuration structure in `docs/admin/config/`

## Testing Guidelines

### Test Organization
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test complete processing pipelines
- **GPU Tests**: Mark with `@pytest.mark.gpu` decorator
- **Mock External Dependencies**: Use pytest mocks for external services

### Test Environment Setup
```python
# Example GPU test structure
@pytest.mark.gpu
def test_gpu_processing():
    import cudf  # Only import in GPU tests
    df = cudf.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    assert len(df) == 3
```

### Test Configuration
- Tests use a unified Ray cluster configuration via `conftest.py`
- GPU availability is automatically detected using multiple methods (pynvml, nvidia-ml-py)
- Tests gracefully handle missing GPU dependencies

### Running Tests
```bash
# Run all tests (CPU only by default)
uv run pytest

# Run GPU tests (requires CUDA environment)
uv run pytest -m gpu

# Run specific test categories
uv run pytest -m "not gpu"  # CPU tests only

# Run tests for specific modules
uv run pytest tests/stages/text/
uv run pytest tests/stages/image/
```

## Build and Development

### Development Workflow
```bash
# Install development dependencies
uv sync

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest

# Run specific test modules
uv run pytest tests/utils/test_nvcodec_utils.py

# Build documentation (if working on docs)
make docs-html

# Check for dependency updates
uv lock --upgrade
```

### Common Development Patterns

#### Processing Pipeline Structure
```python
# Task-centric pipeline following API design patterns
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.classifiers import FineWebMixtralEduClassifier
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter

# Example workflow following API design
ray_client = RayClient()
ray_client.start()

# Define pipeline stages
read_stage = JsonlReader(input_file_path, files_per_partition=1)
classifier_stage = FineWebMixtralEduClassifier()
write_stage = JsonlWriter(output_file_path)

# Build and run pipeline
pipeline = Pipeline(name="classifier_pipeline")
pipeline.add_stage(read_stage)
pipeline.add_stage(classifier_stage)
pipeline.add_stage(write_stage)

result = pipeline.run()
ray_client.stop()
```

#### GPU vs CPU Handling
```python
# Resource specification for stages
from nemo_curator.stages.resources import Resources

class MyProcessingStage(ProcessingStage):
    @property
    def resources(self) -> Resources:
        return Resources(cpus=1.0, gpu_memory_gb=4.0)
```

## API Design Architecture

### Design Principles

#### Task-Centric Architecture
The design of NeMo Curator is based on Ray and operates on individual **Tasks** - batches of data that flow through the pipeline. This enables:
- Finer-grained control and monitoring
- Better resource utilization

#### Map-style (Data-Parallel) Execution
All stages are designed to be map-style on tasks, meaning they take task as input and produce task as output. This allows for easy parallelization and scaling.
- We do not enforce 1-1 mapping between input and output tasks, but rather allow for multiple output tasks from a single input task and multiple input tasks from a single output task. More specifically, a stage applies a transformation from `X` to `Y`, where both `X` and `Y` can be `Task | list[Task] | None`.

#### Fault Tolerance Requirements
**All stages MUST be fault-tolerant and retry-safe.** This is a critical requirement because:

- **Task Preemption:** Xenna can preempt/kill running tasks before completion and potentially reschedule them later, especially during autoscaling events
- **Partial Operations:** Tasks may be interrupted mid-execution, leaving partial state (e.g., incomplete file downloads)

## Core Components

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

### API Status
**Status:** Pre Release - This API design is currently under development and may change.

### Examples and Usage
For practical examples of the API in action, refer to the quickstart examples in `nemo_curator/examples/quickstart.py` and the tutorial notebooks that demonstrate complete pipeline workflows following these design patterns.

## File Structure Conventions

- **Processing Stages**: `nemo_curator/stages/{modality}/` (text, image, audio, video)
- **Utilities**: `nemo_curator/utils/` (shared functionality)
- **Tests**: `tests/` (mirrors source structure with 162+ test files)
- **Documentation**: `docs/` (Sphinx-based documentation with MyST)
- **Examples**: `tutorials/` (Jupyter notebooks and example scripts)
- **Configuration**: `config/` (YAML configuration templates)

### Key Directories
```
nemo_curator/
├── stages/
│   ├── text/          # Text processing pipelines
│   ├── image/         # Image processing and filtering
│   ├── audio/         # Audio processing capabilities
│   └── video/         # Video processing and analysis
├── utils/             # Shared utilities and helpers
├── datasets/          # Dataset loading and manipulation
└── modules/           # Core processing algorithms
```

## GPU Development Notes

### Memory Management
- Use context managers for GPU memory allocation
- Implement memory pool management for large datasets
- Monitor GPU memory usage during development

### CUDA Compatibility
- Ensure CUDA 12.x compatibility for all GPU code
- Test both single-GPU and multi-GPU scenarios
- Handle graceful degradation when GPU resources are unavailable

### Performance Considerations
- Profile GPU kernels for performance bottlenecks
- Use async processing where possible with Ray
- Implement batch processing for efficient GPU utilization

## Documentation Guidelines

- Update docstrings for all public APIs
- Include usage examples in docstrings
- Update relevant documentation in `docs/` for significant changes
- Use type hints and document parameter types and return values

This repository follows NVIDIA's coding standards and emphasizes scalable, robust data processing pipelines with optional GPU acceleration for high-performance computing environments.