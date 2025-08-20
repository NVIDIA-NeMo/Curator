(admin-config-environment-variables)=
# Environment Variables Reference

This comprehensive reference covers all environment variables used by NeMo Curator for runtime configuration, performance optimization, and system integration. Environment variables provide the highest precedence in the configuration hierarchy.

```{tip}
**Applying Environment Variables**: These variables are used throughout NeMo Curator deployments:
- {doc}`Deployment Environment Configuration <deployment-environments>`: Environment-specific variable patterns
- {doc}`Kubernetes Deployment <../deployment/kubernetes>`: Setting variables in Kubernetes ConfigMaps
- {doc}`Slurm Deployment <../deployment/slurm/index>`: Using variables in Slurm job scripts
```

---

## Core NeMo Curator Variables

### Device and Processing Configuration

```{list-table} Device Configuration Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `DEVICE`
  - "cpu"
  - Processing device: "cpu" or "gpu"
* - `INTERFACE`
  - "eth0"
  - Network interface for Dask communication
* - `PROTOCOL`
  - "tcp"
  - Network protocol: "tcp" or "ucx"
* - `CPU_WORKER_MEMORY_LIMIT`
  - "0"
  - Memory limit per CPU worker ("0" = no limit)
```

**Example Usage:**
```bash
# GPU processing with UCX protocol
export DEVICE="gpu"
export PROTOCOL="ucx"
export INTERFACE="ib0"  # InfiniBand interface
export CPU_WORKER_MEMORY_LIMIT="8GB"
```

### Logging and Profiling

```{list-table} Logging Configuration Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `LOGDIR`
  - "./logs"
  - Directory for log files
* - `PROFILESDIR`
  - "./profiles"
  - Directory for performance profiles
* - `SCHEDULER_FILE`
  - auto-generated
  - Path to Dask scheduler connection file
* - `SCHEDULER_LOG`
  - auto-generated
  - Path to scheduler log file
* - `DONE_MARKER`
  - auto-generated
  - Path to job completion marker file
```

**Example Usage:**
```bash
# Custom logging configuration
export LOGDIR="/shared/logs/nemo_curator"
export PROFILESDIR="/shared/profiles"
export SCHEDULER_LOG="/shared/logs/scheduler.log"
```

---

## RAPIDS and GPU Configuration

### Memory Management (RMM)

```{list-table} RMM Memory Pool Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `RMM_WORKER_POOL_SIZE`
  - "72GiB"
  - GPU memory pool size per worker
* - `RMM_SCHEDULER_POOL_SIZE`
  - "1GB"
  - GPU memory pool size for scheduler
* - `RMM_ALLOCATOR`
  - "pool"
  - Memory allocator: "pool", "arena", "binning"
* - `RMM_POOL_INIT_SIZE`
  - "256MB"
  - Initial pool size
* - `RMM_MAXIMUM_POOL_SIZE`
  - auto-detect
  - Maximum pool size
```

**Memory Sizing Guidelines:**
```bash
# For 80GB GPU (A100/H100)
export RMM_WORKER_POOL_SIZE="72GiB"  # 90% of GPU memory

# For 40GB GPU (A100)
export RMM_WORKER_POOL_SIZE="36GiB"  # 90% of GPU memory

# For 16GB GPU (V100)
export RMM_WORKER_POOL_SIZE="14GiB"  # 87.5% of GPU memory

# Percentage-based allocation (alternative)
export RMM_WORKER_POOL_SIZE="0.9"  # 90% of available memory
```

### RAPIDS Initialization

```{list-table} RAPIDS Initialization Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `RAPIDS_NO_INITIALIZE`
  - "1"
  - Delay CUDA context creation: "0" or "1"
* - `CUDF_SPILL`
  - "1"
  - Enable automatic GPU memory spilling: "0" or "1"
* - `CUDF_SPILL_DEVICE_LIMIT`
  - "0.8"
  - Spill threshold (fraction of GPU memory)
* - `LIBCUDF_CUFILE_POLICY`
  - "OFF"
  - GPUDirect Storage policy: "OFF", "ON", "GDS"
```

**Configuration Examples:**
```bash
# High-performance setup (sufficient GPU memory)
export RAPIDS_NO_INITIALIZE="0"  # Initialize immediately
export CUDF_SPILL="0"            # Disable spilling
export LIBCUDF_CUFILE_POLICY="ON"  # Enable direct storage access

# Memory-constrained setup
export RAPIDS_NO_INITIALIZE="1"  # Delay initialization
export CUDF_SPILL="1"            # Enable spilling
export CUDF_SPILL_DEVICE_LIMIT="0.7"  # Spill at 70% capacity
```

---

## Dask Configuration

### Distributed Computing

```{list-table} Dask Distributed Variables
:header-rows: 1
:widths: 40 20 40

* - Variable
  - Default
  - Description
* - `DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT`
  - "10s"
  - Connection timeout
* - `DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP`
  - "30s"
  - TCP timeout
* - `DASK_DISTRIBUTED__WORKER__DAEMON`
  - "True"
  - Run workers as daemons
* - `DASK_DISTRIBUTED__WORKER__MEMORY__TARGET`
  - "0.6"
  - Target memory usage fraction
* - `DASK_DISTRIBUTED__WORKER__MEMORY__SPILL`
  - "0.7"
  - Spill to disk threshold
* - `DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE`
  - "0.8"
  - Pause computation threshold
* - `DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE`
  - "0.95"
  - Terminate worker threshold
```

### Performance Profiling

```{list-table} Dask Profiling Variables
:header-rows: 1
:widths: 40 20 40

* - Variable
  - Default
  - Description
* - `DASK_DISTRIBUTED__WORKER__PROFILE__ENABLED`
  - "False"
  - Enable worker profiling
* - `DASK_DISTRIBUTED__WORKER__PROFILE__INTERVAL`
  - "10ms"
  - Profiling sample interval
* - `DASK_DISTRIBUTED__WORKER__PROFILE__CYCLE`
  - "1000ms"
  - Profiling cycle duration
```

### DataFrame Configuration

```{list-table} Dask DataFrame Variables
:header-rows: 1
:widths: 40 20 40

* - Variable
  - Default
  - Description
* - `DASK_DATAFRAME__CONVERT_STRING`
  - "False"
  - Convert strings to categorical
* - `DASK_DATAFRAME__QUERY_PLANNING`
  - "False"
  - Enable query planning optimization
* - `DASK_DATAFRAME__PARQUET__MINIMUM_PARTITION_SIZE`
  - "128MB"
  - Minimum partition size for Parquet
* - `DASK_DATAFRAME__PARQUET__MAXIMUM_PARTITION_SIZE`
  - "256MB"
  - Maximum partition size for Parquet
* - `DASK_DATAFRAME__PARQUET__COMPRESSION`
  - "snappy"
  - Compression algorithm for Parquet
```

**Optimized DataFrame Settings:**
```bash
# High-performance I/O
export DASK_DATAFRAME__PARQUET__MINIMUM_PARTITION_SIZE="256MB"
export DASK_DATAFRAME__PARQUET__MAXIMUM_PARTITION_SIZE="1GB"
export DASK_DATAFRAME__PARQUET__COMPRESSION="lz4"
export DASK_DATAFRAME__CONVERT_STRING="False"
```

---

## Network and Communication

### UCX Configuration

```{list-table} UCX Communication Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `UCX_TLS`
  - auto-detect
  - Transport layers: "rc,cuda_copy,cuda_ipc"
* - `UCX_NET_DEVICES`
  - auto-detect
  - Network devices to use
* - `UCX_MEMTYPE_CACHE`
  - "y"
  - Enable memory type cache
* - `UCX_RNDV_SCHEME`
  - "put_zcopy"
  - Rendezvous protocol scheme
* - `UCX_IB_GPU_DIRECT_RDMA`
  - "yes"
  - Enable GPU Direct RDMA
```

**InfiniBand Optimization:**
```bash
# Optimized UCX for InfiniBand + GPU
export UCX_TLS="rc,cuda_copy,cuda_ipc"
export UCX_NET_DEVICES="mlx5_0:1"  # Specific InfiniBand device
export UCX_IB_GPU_DIRECT_RDMA="yes"
export UCX_MEMTYPE_CACHE="y"
```

### TCP Configuration

```{list-table} TCP Network Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `DASK_UCX__CUDA_COPY`
  - "True"
  - Enable CUDA memory copy
* - `DASK_UCX__TCP`
  - "True"
  - Enable TCP transport
* - `DASK_UCX__NVLINK`
  - "True"
  - Enable NVLink transport
* - `DASK_UCX__INFINIBAND`
  - "True"
  - Enable InfiniBand transport
* - `DASK_UCX__RDMACM`
  - "True"
  - Enable RDMA CM
```

---

## Storage and I/O

### Cloud Storage Optimization

#### AWS S3 Variables

```{list-table} AWS S3 Configuration Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `AWS_ACCESS_KEY_ID`
  - none
  - AWS access key identifier
* - `AWS_SECRET_ACCESS_KEY`
  - none
  - AWS secret access key
* - `AWS_DEFAULT_REGION`
  - none
  - Default AWS region
* - `AWS_PROFILE`
  - "default"
  - AWS profile to use
* - `AWS_MAX_ATTEMPTS`
  - "5"
  - Maximum retry attempts
* - `AWS_RETRY_MODE`
  - "legacy"
  - Retry mode: "legacy", "standard", "adaptive"
* - `AWS_S3_USE_ACCELERATE_ENDPOINT`
  - "false"
  - Use S3 Transfer Acceleration
* - `AWS_S3_ADDRESSING_STYLE`
  - "auto"
  - S3 addressing style: "auto", "virtual", "path"
```

#### Azure Storage Variables

```{list-table} Azure Storage Configuration Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `AZURE_STORAGE_CONNECTION_STRING`
  - none
  - Azure storage connection string
* - `AZURE_STORAGE_ACCOUNT_NAME`
  - none
  - Azure storage account name
* - `AZURE_STORAGE_ACCOUNT_KEY`
  - none
  - Azure storage account key
* - `AZURE_STORAGE_SAS_TOKEN`
  - none
  - Azure SAS token
```

### Local I/O Optimization

```{list-table} Local I/O Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `OMP_NUM_THREADS`
  - CPU count
  - OpenMP thread count
* - `MKL_NUM_THREADS`
  - CPU count
  - Intel MKL thread count
* - `NUMBA_NUM_THREADS`
  - CPU count
  - Numba thread count
* - `TMPDIR`
  - "/tmp"
  - Temporary directory
* - `PYTHONPATH`
  - system default
  - Python module search path
```

**Thread Optimization:**
```bash
# Prevent oversubscription in distributed environments
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export NUMBA_NUM_THREADS="1"

# Use fast storage for temporary files
export TMPDIR="/fast/ssd/tmp"
```

---

## API and Service Configuration

### Machine Learning Services

```{list-table} ML Service API Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `HUGGINGFACE_HUB_TOKEN`
  - none
  - HuggingFace Hub API token
* - `HF_HOME`
  - "~/.cache/huggingface"
  - HuggingFace cache directory
* - `OPENAI_API_KEY`
  - none
  - OpenAI API key
* - `OPENAI_ORG`
  - none
  - OpenAI organization ID
* - `NVIDIA_API_KEY`
  - none
  - NVIDIA AI foundation models API key
* - `NVIDIA_BASE_URL`
  - "https://integrate.api.nvidia.com/v1"
  - NVIDIA API base URL
```

### Model Caching

```{list-table} Model Cache Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `TRANSFORMERS_CACHE`
  - "~/.cache/huggingface/transformers"
  - Transformers model cache
* - `HF_DATASETS_CACHE`
  - "~/.cache/huggingface/datasets"
  - HuggingFace datasets cache
* - `TORCH_HOME`
  - "~/.cache/torch"
  - PyTorch model cache
* - `SENTENCE_TRANSFORMERS_HOME`
  - "~/.cache/torch/sentence_transformers"
  - Sentence transformers cache
```

---

## CUDA and GPU Runtime

### CUDA Configuration

```{list-table} CUDA Runtime Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `CUDA_VISIBLE_DEVICES`
  - all
  - Visible GPU devices (e.g., "0,1,2,3")
* - `CUDA_LAUNCH_BLOCKING`
  - "0"
  - Synchronous CUDA launches: "0" or "1"
* - `CUDA_CACHE_PATH`
  - auto
  - CUDA kernel cache path
* - `CUDA_FORCE_PTX_JIT`
  - "0"
  - Force PTX JIT compilation
* - `CUDA_MODULE_LOADING`
  - "LAZY"
  - Module loading strategy: "LAZY" or "EAGER"
```

### GPU Memory Management

```{list-table} GPU Memory Variables
:header-rows: 1
:widths: 30 20 50

* - Variable
  - Default
  - Description
* - `PYTORCH_CUDA_ALLOC_CONF`
  - none
  - PyTorch CUDA allocator configuration
* - `CUDA_MPS_PIPE_DIRECTORY`
  - none
  - Multi-Process Service pipe directory
* - `CUDA_MPS_LOG_DIRECTORY`
  - none
  - Multi-Process Service log directory
```

**Memory Optimization Examples:**
```bash
# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Enable CUDA MPS for multi-process GPU sharing
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log"
```

---

## Environment Variable Profiles

### Development Profile

```bash
# Development environment variables
export DEVICE="cpu"
export PROTOCOL="tcp"
export INTERFACE="eth0"
export CPU_WORKER_MEMORY_LIMIT="4GB"
export LOGDIR="./dev_logs"
export PROFILESDIR="./dev_profiles"
export DASK_DISTRIBUTED__WORKER__PROFILE__ENABLED="True"
export OMP_NUM_THREADS="2"
export MKL_NUM_THREADS="2"
```

### Production CPU Profile

```bash
# Production CPU environment
export DEVICE="cpu"
export PROTOCOL="tcp"
export INTERFACE="eth0"
export CPU_WORKER_MEMORY_LIMIT="0"  # No limit
export LOGDIR="/shared/logs"
export PROFILESDIR="/shared/profiles"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export DASK_DATAFRAME__PARQUET__MINIMUM_PARTITION_SIZE="256MB"
export AWS_MAX_ATTEMPTS="10"
export AWS_RETRY_MODE="adaptive"
```

### Production GPU Profile

```bash
# Production GPU environment
export DEVICE="gpu"
export PROTOCOL="ucx"
export INTERFACE="ib0"
export RAPIDS_NO_INITIALIZE="0"
export CUDF_SPILL="0"
export RMM_WORKER_POOL_SIZE="72GiB"
export RMM_SCHEDULER_POOL_SIZE="1GB"
export LIBCUDF_CUFILE_POLICY="ON"
export UCX_TLS="rc,cuda_copy,cuda_ipc"
export UCX_IB_GPU_DIRECT_RDMA="yes"
export LOGDIR="/shared/logs"
export PROFILESDIR="/shared/profiles"
```

### Memory-Constrained Profile

```bash
# Memory-constrained environment
export DEVICE="gpu"
export PROTOCOL="tcp"
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"
export CUDF_SPILL_DEVICE_LIMIT="0.7"
export RMM_WORKER_POOL_SIZE="12GB"  # Smaller pool
export CPU_WORKER_MEMORY_LIMIT="8GB"
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET="0.5"
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL="0.6"
```

---

## Environment Variable Management

### Loading Environment Variables

#### From File

```bash
# Load from environment file
set -a  # Automatically export variables
source /path/to/nemo-curator.env
set +a  # Stop auto-export

# Or use explicit loading
export $(cat /path/to/nemo-curator.env | xargs)
```

#### Systemd Service

```ini
# /etc/systemd/system/nemo-curator.service
[Unit]
Description=NeMo Curator Service
After=network.target

[Service]
Type=exec
User=curator
Group=curator
EnvironmentFile=/etc/nemo-curator/environment
ExecStart=/usr/local/bin/nemo-curator-script
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

#### Docker Environment

```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/nemo:latest

# Set environment variables
ENV DEVICE=gpu
ENV PROTOCOL=ucx
ENV RMM_WORKER_POOL_SIZE=72GiB
ENV CUDF_SPILL=0

# Or load from file
COPY nemo-curator.env /etc/environment
RUN set -a && source /etc/environment && set +a
```

### Validation Script

```python
#!/usr/bin/env python3
"""Validate NeMo Curator environment variables."""

import os
import sys

def validate_environment():
    """Validate environment variable configuration."""
    
    required_vars = {
        "DEVICE": ["cpu", "gpu"],
        "PROTOCOL": ["tcp", "ucx"],
    }
    
    recommended_vars = {
        "LOGDIR": str,
        "RMM_WORKER_POOL_SIZE": str,
        "CUDF_SPILL": ["0", "1"],
    }
    
    issues = []
    
    # Check required variables
    for var, valid_values in required_vars.items():
        value = os.getenv(var)
        if not value:
            issues.append(f"Missing required variable: {var}")
        elif valid_values and value not in valid_values:
            issues.append(f"Invalid value for {var}: {value} (valid: {valid_values})")
    
    # Check recommended variables
    for var, expected_type in recommended_vars.items():
        value = os.getenv(var)
        if not value:
            print(f"⚠ Recommended variable not set: {var}")
        elif expected_type == str:
            print(f"✓ {var} = {value}")
        elif isinstance(expected_type, list) and value not in expected_type:
            issues.append(f"Invalid value for {var}: {value} (valid: {expected_type})")
    
    # GPU-specific validation
    if os.getenv("DEVICE") == "gpu":
        gpu_vars = ["RMM_WORKER_POOL_SIZE", "CUDF_SPILL"]
        for var in gpu_vars:
            if not os.getenv(var):
                issues.append(f"GPU mode requires {var} to be set")
    
    # Report results
    if issues:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Environment validation passed")
        return True

if __name__ == "__main__":
    success = validate_environment()
    sys.exit(0 if success else 1)
```
