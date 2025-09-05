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
- **Text Processing**: BeautifulSoup, fasttext, sentencepiece, trafilatura
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
# Graceful degradation for optional GPU dependencies
try:
    import cudf
    import cvcuda
    import PyNvVideoCodec as Nvc
    HAS_GPU_SUPPORT = True
except ImportError:
    HAS_GPU_SUPPORT = False
    # Provide helpful error messages for missing dependencies
    logger.warning("GPU dependencies not available, falling back to CPU processing")

# Example from nvcodec_utils.py
try:
    import cvcuda
    import nvcv
    import pycuda.driver as cuda
    import PyNvVideoCodec as Nvc
    pixel_format_to_cvcuda_code = {
        Nvc.Pixel_Format.YUV444: cvcuda.ColorConversion.YUV2RGB,
        Nvc.Pixel_Format.NV12: cvcuda.ColorConversion.YUV2RGB_NV12,
    }
except (ImportError, RuntimeError):
    logger.warning("PyNvVideoCodec is not installed, some features will be disabled.")
    Nvc = None
    pixel_format_to_cvcuda_code = {}
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
# Standard pipeline component structure
class ProcessingStage:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        # Process and return modified dataset
        pass
```

#### GPU vs CPU Handling
```python
# Check for GPU availability and fallback gracefully
if HAS_GPU_SUPPORT and use_gpu:
    # Use GPU-accelerated processing
    df = cudf.DataFrame(data)
else:
    # Fallback to CPU processing
    df = pd.DataFrame(data)
```

#### Configuration Loading
```python
# Use hierarchical configuration loading
from nemo_curator.utils.config_utils import load_config

config = load_config(
    config_path="config/processing.yaml",
    overrides={"batch_size": 1024}
)
```

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