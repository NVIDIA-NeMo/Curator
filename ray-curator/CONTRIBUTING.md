> [!note]
> This document is still a work in progress and may change frequently.

## Setup and Dev

### Prerequisites

- Python >=3.10, < 3.13
- OS: Ubuntu 22.04/20.04
- NVIDIA GPU (optional)
  - Volta™ or higher (compute capability 7.0+)
  - CUDA 12.x

### Installation

1. Standard Installation
    ```bash
    # From the root of the NeMo-Curator repository
    cd ray-curator
    pip install --extra-index-url https://pypi.nvidia.com .
    ```
2. Dev Installation (with testing dependencies and editable install)

    ```
    cd ray-curator
    pip install --extra-index-url https://pypi.nvidia.com -e ".[dev]"
    ```

### Dev Pattern

- All upstream work/changes with the new API and ray backend should target the `NeMo-Curator/ray-api` branch.
- When re-using code already in `NeMo-Curator/nemo_curator`, use `git mv` to move those source files into the `ray-curator/ray_curator` namespace.
- Sign and signoff commits with `git commit -sS`. (May be relaxed in the future)

### Testing

Work in Progress...
