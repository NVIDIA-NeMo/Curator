# NVIDIA NeMo Curator
Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

NVIDIA NeMo Curator is a Python library for fast and scalable data processing and curation for generative AI use cases such as foundation language model pretraining, text-to-image model training, domain-adaptive pretraining, supervised fine-tuning, and parameter-efficient fine-tuning. It leverages GPUs with Ray and RAPIDS for significant performance improvements.

## Working Effectively

### Prerequisites and Environment Setup
- **Python Version**: 3.12 ONLY. Python 3.11 is NOT supported due to RAPIDS compatibility issues.
- **Package Manager**: Prefers `uv` but `pip` works as fallback.
- **System**: Ubuntu 22.04/20.04 recommended for production.
- **GPUs**: Optional but recommended for performance. CUDA 12.x support available.

### Bootstrap, Build, and Test the Repository
1. **Install the package**: 
   ```bash
   pip install -e .
   ```
   Takes 2-3 minutes. NEVER CANCEL. Set timeout to 5+ minutes.

2. **Install with specific feature sets** (choose based on your needs):
   ```bash
   # Text processing capabilities
   pip install -e ".[text]"
   
   # GPU deduplication (requires CUDA)
   pip install -e ".[deduplication_cuda12x]"
   
   # Video processing 
   pip install -e ".[video]"
   
   # All features
   pip install -e ".[all]"
   ```
   Takes 5-10 minutes depending on features. NEVER CANCEL. Set timeout to 15+ minutes.

3. **Run tests**:
   ```bash
   # CPU tests only (recommended for development)
   python3 -m pytest tests/ -v -m "not gpu"
   
   # All tests (requires GPU)
   python3 -m pytest tests/ -v
   ```
   CPU tests take ~2-3 minutes. NEVER CANCEL. Set timeout to 10+ minutes.

4. **Install development dependencies**:
   ```bash
   pip install pytest pytest-coverage ruff==0.11.4
   ```

### Linting and Code Quality
- **Always run linting before committing**:
  ```bash
  ruff check .
  ruff format .
  ```
  Takes 5-15 seconds. Clean codebase should show no errors.

### Documentation
- **Build documentation**:
  ```bash
  # Create docs environment first
  python3 -m venv .venv-docs
  source .venv-docs/bin/activate
  pip install -r requirements-docs.txt
  
  # Build docs (use existing deps if available)
  python -m sphinx -b html docs docs/_build/html
  ```
  Takes 30-60 seconds. NEVER CANCEL. Set timeout to 3+ minutes.

- **Use Makefile shortcuts** (requires .venv-docs setup):
  ```bash
  make docs-html          # Standard build
  make docs-live          # Live reload server
  make docs-html-ga       # GA variant
  ```

## Validation Scenarios

### Always Test These Core Functionalities After Making Changes
1. **Basic import validation**:
   ```python
   import nemo_curator
   print("✓ NeMo Curator imported successfully")
   
   # Test key components
   from nemo_curator.pipeline import Pipeline
   from nemo_curator.stages.base import ProcessingStage  
   from nemo_curator.tasks import Task
   print("✓ Key components imported successfully")
   
   # Test pipeline creation
   pipeline = Pipeline(name="test", description="test pipeline")
   print(f"✓ Pipeline created: {pipeline.name}")
   ```

2. **Validate Ray functionality**:
   ```python
   from nemo_curator.core.client import RayClient
   client = RayClient()
   client.start()
   print("✓ Ray started successfully")
   client.stop()
   print("✓ Ray stopped successfully")
   ```

3. **Test backends without GPU** (comprehensive test):
   ```bash
   python3 -m pytest tests/backends/experimental/ray_data/test_utils.py -v -m "not gpu"
   ```
   Should pass 13 tests in ~7 seconds.

4. **Run quickstart example** (TODO: GPU functionality not available yet, focus on CPU examples):
   ```bash
   python examples/quickstart.py
   ```
   Should show pipeline description, start Ray cluster, then fail gracefully with GPU resource message.

5. **Verify linting setup**:
   ```bash
   ruff check nemo_curator/ --quiet
   ```
   Should complete in <1 second with no output (clean code).

## Critical Timing and Resource Information

### NEVER CANCEL Commands
- **Package installation**: 2-15 minutes depending on features
- **Full test suite**: 5-15 minutes 
- **Documentation build**: 1-3 minutes
- **GPU tests**: 10-30 minutes (if available)

### Timeout Recommendations
- Basic pip install: 10+ minutes
- Tests with GPU marker: 30+ minutes  
- Documentation builds: 5+ minutes
- CI pipeline: 45+ minutes

## Common Development Tasks

### Working with Text Curation (Most Common)
- **Pipeline stages**: Download → Process → Generate
- **Key modules**: `nemo_curator.stages.text.*`
- **Example workflows**: See `examples/` and `tutorials/`

### Working with Image Curation
- **Pipeline stages**: Load → Process
- **Key modules**: `nemo_curator.stages.image.*`
- **Requires**: WebDataset format inputs

### Working with GPU Acceleration
- **Check GPU availability**: Test scripts will detect automatically
- **RAPIDS required**: For fuzzy deduplication, semantic deduplication
- **Fallback**: CPU-only mode available for most operations

### Key Project Structure
```
nemo_curator/           # Main library code
├── backends/           # Execution backends (Ray, Xenna)
├── stages/            # Processing stages
│   ├── text/          # Text processing
│   ├── image/         # Image processing
│   └── video/         # Video processing
├── pipeline/          # Pipeline orchestration
└── tasks/             # Task definitions

tests/                 # Test suite with GPU/CPU markers
docs/                  # Sphinx documentation
examples/              # Working examples
tutorials/             # Educational content
```

### Troubleshooting Common Issues
- **Python 3.11 error**: Use Python 3.12 only
- **RAPIDS installation fails**: Install with all features: `pip install -e ".[all]"`
- **GPU not found**: Most functionality works CPU-only, examples gracefully degrade
- **Network timeouts during install**: Use `--timeout 600` with pip
- **Documentation build fails**: Ensure .venv-docs is properly set up

### Pre-commit and CI Requirements
- **Pre-commit hooks**: Run `ruff check .` and `ruff format .`
- **Sign commits**: Use `git commit -s` for signed-off commits
- **CI expectations**: CPU tests must pass, GPU tests optional in CI

## Performance Context
- **16× faster fuzzy deduplication** on 8TB datasets with GPU acceleration
- **Near-linear scaling** across 1-4 H100 80GB nodes  
- **~40% lower TCO** compared to CPU-only approaches
- **Primary bottleneck**: GPU memory for large-scale operations

Always check the latest README.md and CONTRIBUTING.md for any updates to these workflows.