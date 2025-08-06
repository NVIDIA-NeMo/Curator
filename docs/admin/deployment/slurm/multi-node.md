---
description: "Advanced multi-node Slurm configurations for large-scale NeMo Curator deployments with performance optimization and troubleshooting"
categories: ["how-to-guides"]
tags: ["multi-node", "slurm", "advanced", "performance", "scaling", "networking", "troubleshooting", "production"]
personas: ["admin-focused", "devops-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "universal"
---

(admin-deployment-slurm-multi-node)=
# Multi-Node Slurm Setup Guide

This guide covers advanced multi-node Slurm configurations for running NeMo-Curator at scale across 4+ nodes, with focus on performance optimization, network configuration, and production deployment patterns.

```{seealso}
For basic Slurm deployment, see [Deploy All Modalities on Slurm](admin-deployment-slurm-general). This guide assumes familiarity with the basic concepts covered there.
```

## Getting Started: Step-by-Step Workflow

This section provides a procedural approach to implementing multi-node NeMo-Curator deployments.

### Assessment and Planning

1.  Assess Your Infrastructure.
    ```bash
    # Run this assessment script on your cluster
    ./infrastructure_assessment.sh

    #!/bin/bash
    # infrastructure_assessment.sh
    echo "=== Cluster Assessment for NeMo-Curator Multi-Node ==="

    echo "1. Node Information:"
    sinfo -N -l

    echo -e "\n2. Available Partitions:"
    sinfo -s

    echo -e "\n3. Network Interfaces (run on compute node):"
    srun --nodes=1 --pty bash -c "
        echo 'Available interfaces:'
        ip link show | grep -E '^[0-9]+:' | awk '{print \$2}' | sed 's/://'
        echo 'Interface speeds:'
        for iface in \$(ip link show | grep -E '^[0-9]+:' | awk '{print \$2}' | sed 's/://'); do
            if [[ \$iface != 'lo' ]]; then
                speed=\$(cat /sys/class/net/\$iface/speed 2>/dev/null || echo 'unknown')
                echo \"\$iface: \${speed} Mbps\"
            fi
        done
    "

    echo -e "\n4. Shared Filesystem Check:"
    df -h | grep -E "(nfs|lustre|gpfs|fhgfs)"

    echo -e "\n5. Container Runtime:"
    which singularity || which apptainer || echo "No container runtime found"
    ```

2. Determine Your Configuration. Answer these questions to choose your setup:

   | Question | Answer | Recommended Configuration |
   |----------|---------|--------------------------|
   | Dataset size? | <500GB | 4 nodes |
   | | 500GB-2TB | 8 nodes |
   | | >2TB | 16+ nodes |
   | Processing type? | CPU-only | TCP protocol |
   | | GPU + InfiniBand | UCX protocol |
   | | GPU + Ethernet | TCP protocol |
   | Experience level? | Beginner | Start with 4 nodes |
   | | Advanced | Use templates directly |

### Initial Setup and Testing

1. Create Your Working Directory.

   ```bash
   # Create directory structure
   mkdir -p ~/nemo-curator-multinode/{scripts,configs,logs,jobs}
   cd ~/nemo-curator-multinode

   # Copy base templates
   cp /path/to/examples/slurm/start-slurm.sh scripts/
   cp /path/to/examples/slurm/container-entrypoint.sh scripts/
   ```

2. Run Network Discovery. 

   ```bash
   # Create and run network discovery script
   cat > scripts/network_discovery.sh << 'EOF'
   #!/bin/bash
   echo "=== Network Discovery ==="
   echo "Available interfaces:"
   ip link show | grep -E '^[0-9]+:' | awk '{print $2}' | sed 's/://'

   echo -e "\nRecommended interface:"
   for pattern in "ib" "mlx" "25g" "10g" "eth"; do
       iface=$(ip link show | grep -i $pattern | head -1 | awk '{print $2}' | sed 's/://')
       if [[ -n $iface ]]; then
           echo "Use: $iface (type: $pattern)"
           break
       fi
   done
   EOF

   chmod +x scripts/network_discovery.sh
   srun --nodes=1 --pty scripts/network_discovery.sh
   ```

3. Test Basic 2-Node Setup. 

   ```bash
   # Create minimal test job
   cat > jobs/test_2node.sh << 'EOF'
   #!/bin/bash
   #SBATCH --job-name=nemo-curator-test-2node
   #SBATCH --nodes=2
   #SBATCH --exclusive
   #SBATCH --time=00:30:00

   # Update these paths
   export CONTAINER_IMAGE="/path/to/your/container.sqsh"
   export INTERFACE="eth0"  # Update from network discovery
   export PROTOCOL="tcp"
   export CACHE_DIR="$PWD/test-cache"  # Add cache directory

   # Test Dask cluster startup
   export BASE_JOB_DIR=$PWD/test-jobs
   export JOB_DIR=$BASE_JOB_DIR/$SLURM_JOB_ID
   export LOGDIR=$JOB_DIR/logs
   export SCHEDULER_FILE=$LOGDIR/scheduler.json
   export DONE_MARKER=$LOGDIR/done.txt

   # Simple test command
   export SCRIPT_COMMAND="python -c 'from dask.distributed import Client; c=Client(scheduler_file=\"$SCHEDULER_FILE\"); print(f\"Workers: {len(c.scheduler_info()[\"workers\"])}\"); c.close()'"

   # Use basic entrypoint for testing
   srun --container-image=${CONTAINER_IMAGE} /path/to/container-entrypoint.sh
   EOF

   # Submit test job
   sbatch jobs/test_2node.sh
   ```

### Scale to Multi-Node

1.  Based on your assessment, copy the appropriate template:

    ```bash
    # Start with the basic Slurm script and customize it
    cp examples/slurm/start-slurm.sh jobs/my_text_job.sh

    # Or use the existing container entrypoint as reference
    cp examples/slurm/container-entrypoint.sh scripts/my-entrypoint.sh
    ```

2. Configure Your Job. Edit the template and update all "Update Me!" sections:

   ```bash
   # Edit your job script
   nano jobs/my_text_job.sh

   # Required updates:
   # - Container image path
   # - Input/output data paths
   # - Network interface (from discovery)
   # - Resource allocation
   # - Script command for your specific task
   ```

3.  Create Multi-Node Entrypoint.

    ```bash
    # Copy the advanced entrypoint script
    cp scripts/multi-node-entrypoint.sh scripts/my-entrypoint.sh

    # Make it executable
    chmod +x scripts/my-entrypoint.sh

    # Update job script to use it
    sed -i 's|CONTAINER_ENTRYPOINT=.*|CONTAINER_ENTRYPOINT='$(pwd)'/scripts/my-entrypoint.sh|' jobs/my_text_job.sh
    ```

### Deployment and Monitoring

1. Deploy with Monitoring.

   ```bash
   # Submit your multi-node job
   JOB_ID=$(sbatch --parsable jobs/my_text_job.sh)
   echo "Submitted job: $JOB_ID"

   # Monitor the job
   watch squeue -j $JOB_ID

   # Monitor logs in real-time
   tail -f logs/nemo-curator-*_${JOB_ID}.log
   ```

2. Set Up Cluster Monitoring.

   ```bash
   # Create monitoring script
   cat > scripts/monitor_cluster.sh << 'EOF'
   #!/bin/bash
   JOB_ID=$1
   SCHEDULER_FILE=$PWD/test-jobs/$JOB_ID/logs/scheduler.json

   if [[ -f $SCHEDULER_FILE ]]; then
       python -c "
   from dask.distributed import Client
   client = Client(scheduler_file='$SCHEDULER_FILE', timeout='10s')
   info = client.scheduler_info()
   workers = info['workers']
   print(f'Active Workers: {len(workers)}')
   print(f'Total Cores: {sum(w[\"nthreads\"] for w in workers.values())}')
   total_mem = sum(w['memory_limit'] for w in workers.values()) / 1e9
   used_mem = sum(w['metrics']['memory'] for w in workers.values()) / 1e9
   print(f'Memory: {used_mem:.1f}GB / {total_mem:.1f}GB')
   client.close()
   "
   else
       echo "Scheduler file not found: $SCHEDULER_FILE"
   fi
   EOF

   chmod +x scripts/monitor_cluster.sh

   # Usage
   ./scripts/monitor_cluster.sh $JOB_ID
   ```

### Optimization and Scaling (As needed)

1. Performance Tuning.

   ```bash
   # After successful runs, optimize based on logs
   grep -i "memory\|warning\|error" logs/nemo-curator-*_${JOB_ID}.log

   # Adjust memory settings if needed
   # Adjust worker counts if CPU underutilized
   # Adjust network settings if communication slow
   ```

2. Scale Up.

   ```bash
   # Test scaling efficiency
   for nodes in 4 8 16; do
       echo "Testing $nodes nodes..."
       sed "s/#SBATCH --nodes=.*/#SBATCH --nodes=$nodes/" jobs/my_text_job.sh > jobs/scale_test_${nodes}n.sh
       sbatch jobs/scale_test_${nodes}n.sh
   done
   ```

## Quick Start Checklist

For experienced users, here's a rapid deployment checklist:

- [ ] **Infrastructure Ready**: Shared filesystem, container runtime, Slurm access
- [ ] **Network Discovered**: Run network discovery, note optimal interface
- [ ] **Template Selected**: Choose 4-node, 8-node, or 16-node template
- [ ] **Paths Updated**: Container image, data paths, interface in job script  
- [ ] **Test Job**: Submit 2-node test to verify basic functionality
- [ ] **Production Job**: Submit multi-node job with monitoring
- [ ] **Optimize**: Tune memory, workers, network based on performance

## Troubleshooting Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| Workers not connecting | Check network interface, firewall rules |
| Out of memory errors | Reduce workers per node, increase memory limits |
| Slow inter-node communication | Switch to UCX protocol, check network bandwidth |
| Container mount errors | Verify shared filesystem paths |
| Job stuck in queue | Check partition limits, resource requests |

## Prerequisites

### Hardware Requirements

- **Multi-node cluster**: 4+ compute nodes (recommended for this guide)
- **Network**: High-bandwidth interconnect (InfiniBand recommended for 8+ nodes)
- **Storage**: Parallel filesystem (Lustre, GPFS, or high-performance NFS)
- **Memory**: 256GB+ RAM per node for large-scale text processing
- **GPUs**: 4-8 GPUs per node for GPU-accelerated workloads

### Software Requirements

- Slurm workload manager with multi-node job support
- Container runtime (Singularity/Apptainer) with multi-node networking
- Shared filesystem mounted consistently across all nodes
- Network configuration allowing inter-node Dask communication

## Network Configuration

### Interface Detection and Selection

Multi-node deployments require careful network interface selection. Use this script to identify optimal interfaces:

```bash
#!/bin/bash
# network_discovery.sh - Identify best network interfaces

echo "=== Network Interface Discovery ==="
echo "Available interfaces:"
ip link show | grep -E '^[0-9]+:' | awk '{print $2}' | sed 's/://'

echo -e "\n=== Interface Speeds ==="
for iface in $(ip link show | grep -E '^[0-9]+:' | awk '{print $2}' | sed 's/://'); do
    if [[ $iface != "lo" ]]; then
        speed=$(cat /sys/class/net/$iface/speed 2>/dev/null || echo "unknown")
        echo "$iface: ${speed} Mbps"
    fi
done

echo -e "\n=== Recommended Interface Selection ==="
# Prefer InfiniBand, then 25G+, then 10G+, then 1G
for pattern in "ib" "mlx" "25g" "10g" "eth"; do
    iface=$(ip link show | grep -i $pattern | head -1 | awk '{print $2}' | sed 's/://')
    if [[ -n $iface ]]; then
        echo "Recommended: $iface (pattern: $pattern)"
        break
    fi
done
```

### Protocol Selection

Choose the optimal networking protocol based on your infrastructure:

| Infrastructure | Recommended Protocol | Configuration |
|---------------|---------------------|---------------|
| InfiniBand | UCX | `PROTOCOL=ucx` |
| 25G+ Ethernet | UCX or TCP | `PROTOCOL=ucx` (if supported) |
| 10G Ethernet | TCP | `PROTOCOL=tcp` |
| 1G Ethernet | TCP | `PROTOCOL=tcp` |

## Cluster Architecture Planning

### Node Allocation Strategies

**Scheduler Node Strategy:**
```bash
# Option 1: Dedicated scheduler node (recommended for 8+ nodes)
#SBATCH --nodes=9
# Node 0: Scheduler only
# Nodes 1-8: Workers only

# Option 2: Shared scheduler node (suitable for 4-6 nodes)  
#SBATCH --nodes=4
# Node 0: Scheduler + Worker
# Nodes 1-3: Workers only
```

**Resource Distribution:**
```bash
# Calculate optimal worker distribution
calculate_workers() {
    local total_nodes=$1
    local cores_per_node=$2
    local memory_per_node_gb=$3
    
    # Reserve resources for scheduler and system
    local available_cores=$((cores_per_node - 4))
    local available_memory_gb=$((memory_per_node_gb - 16))
    
    # Calculate workers based on memory constraints (8GB per worker recommended)
    local workers_by_memory=$((available_memory_gb / 8))
    local workers_by_cores=$((available_cores / 2))
    
    # Use the more restrictive limit
    if [[ $workers_by_memory -lt $workers_by_cores ]]; then
        echo $workers_by_memory
    else
        echo $workers_by_cores
    fi
}

# Example for 64-core, 512GB nodes
WORKERS_PER_NODE=$(calculate_workers 1 64 512)
echo "Recommended workers per node: $WORKERS_PER_NODE"
```

## Advanced Job Script Templates

### Large-Scale Text Processing (8-Node Template)

```bash
#!/bin/bash
#SBATCH --job-name=nemo-curator-8node
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8

# Performance optimization
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block

# Update Me!
#SBATCH --output=/shared/logs/%x_%j.log
#SBATCH --error=/shared/logs/%x_%j.log
export CONTAINER_IMAGE="/shared/containers/nemo-curator.sqsh"
export INPUT_DATA="/shared/datasets/raw_text"
export OUTPUT_DATA="/shared/datasets/processed_text"
export CACHE_DIR="/shared/cache"
#

# === Multi-node Configuration ===
export BASE_JOB_DIR=/shared/nemo-curator-jobs
export JOB_DIR=$BASE_JOB_DIR/$SLURM_JOB_ID
export LOGDIR=$JOB_DIR/logs
export PROFILESDIR=$JOB_DIR/profiles
export SCHEDULER_FILE=$LOGDIR/scheduler.json
export SCHEDULER_LOG=$LOGDIR/scheduler.log
export DONE_MARKER=$LOGDIR/done.txt

# Auto-detect network interface
export INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)
if ip link show | grep -q "ib"; then
    export INTERFACE=$(ip link show | grep "ib" | head -1 | awk '{print $2}' | sed 's/://')
    export PROTOCOL=ucx
else
    export PROTOCOL=tcp
fi

echo "Using interface: $INTERFACE, protocol: $PROTOCOL"

# === Resource Configuration ===
export DEVICE=gpu
export CPU_WORKERS_PER_NODE=32
export CPU_WORKER_MEMORY_LIMIT=8GB
export GPU_WORKERS_PER_NODE=8

# GPU Memory Configuration
export RAPIDS_NO_INITIALIZE=1
export CUDF_SPILL=1
export RMM_SCHEDULER_POOL_SIZE=2GB
export RMM_WORKER_POOL_SIZE=64GiB
export LIBCUDF_CUFILE_POLICY=KVIKIO

# UCX Configuration for InfiniBand
if [[ $PROTOCOL == "ucx" ]]; then
    export UCX_RNDV_SCHEME=put_zcopy
    export UCX_MEMTYPE_CACHE=n
    export UCX_TLS=rc,ud,mm,shm,cuda_copy,cuda_ipc
fi

# === Script Command ===
export SCRIPT_COMMAND="python -m nemo_curator.scripts.fuzzy_deduplication.compute_minhashes \
    --input-data-dirs $INPUT_DATA \
    --output-minhash-dir $CACHE_DIR/minhashes \
    --scheduler-file $SCHEDULER_FILE \
    --device $DEVICE \
    --input-json-id-field id \
    --char-ngram 24"

# === Container Mounts ===
export MOUNTS="$INPUT_DATA:$INPUT_DATA,$OUTPUT_DATA:$OUTPUT_DATA,$CACHE_DIR:$CACHE_DIR,$JOB_DIR:$JOB_DIR"
export CONTAINER_ENTRYPOINT=/shared/scripts/multi-node-entrypoint.sh

# === Launch Multi-node Job ===
srun \
    --container-mounts=${MOUNTS} \
    --container-image=${CONTAINER_IMAGE} \
    --container-writable \
    ${CONTAINER_ENTRYPOINT}
```

### Multi-Modal Processing (16-Node Template)

```bash
#!/bin/bash
#SBATCH --job-name=nemo-curator-16node-quality-classifier
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8

# Job dependency and coordination
#SBATCH --dependency=singleton
#SBATCH --kill-on-invalid-dep=yes

# === Quality Classification Configuration ===
export BATCH_SIZE=64
export MAX_CHARS=6000

# === Script Command ===
export SCRIPT_COMMAND="python -m nemo_curator.scripts.classifiers.quality_classifier_inference \
    --input-data-dir $INPUT_DATA \
    --output-data-dir $OUTPUT_DATA \
    --scheduler-file $SCHEDULER_FILE \
    --device gpu \
    --batch-size $BATCH_SIZE \
    --max-chars $MAX_CHARS"
```

## Performance Optimization

### Memory Management

**Memory Pool Configuration:**
```bash
# Calculate optimal memory pools based on available resources
calculate_memory_pools() {
    local total_memory_gb=$(free -g | awk 'NR==2{print $2}')
    local system_reserve=32
    local available_memory=$((total_memory_gb - system_reserve))
    
    # Allocate 80% to workers, 20% to system/buffers
    local worker_memory=$((available_memory * 80 / 100))
    local worker_pool_size="${worker_memory}GB"
    
    echo "export CPU_WORKER_MEMORY_LIMIT=${worker_pool_size}"
    echo "export RMM_WORKER_POOL_SIZE=$((worker_memory * 80 / 100))GiB"
}

eval $(calculate_memory_pools)
```

**GPU Memory Optimization:**
```bash
# GPU memory configuration for large-scale processing
export GPU_MEMORY_CONFIG="
    RAPIDS_NO_INITIALIZE=1
    CUDF_SPILL=1
    RMM_WORKER_POOL_SIZE=60GiB
    RMM_SCHEDULER_POOL_SIZE=4GiB
    LIBCUDF_CUFILE_POLICY=KVIKIO
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"
```

### Network Optimization

**Bandwidth Testing:**
```bash
# Test inter-node bandwidth before processing
test_network_bandwidth() {
    echo "=== Network Bandwidth Test ==="
    if command -v iperf3 &> /dev/null; then
        # Run on scheduler node
        if [[ $SLURM_NODEID -eq 0 ]]; then
            iperf3 -s -D  # Start server in background
            sleep 5
        fi
        
        # Test from worker nodes
        if [[ $SLURM_NODEID -gt 0 ]]; then
            scheduler_ip=$(scontrol show hostname $SLURM_JOB_NODELIST | head -1)
            iperf3 -c $scheduler_ip -t 10 -P 4
        fi
    fi
}
```

**UCX Tuning for InfiniBand:**
```bash
# Advanced UCX configuration for high-performance networking
export UCX_PERFORMANCE_CONFIG="
    UCX_RNDV_SCHEME=put_zcopy
    UCX_RNDV_THRESH=8192
    UCX_TLS=rc,ud,mm,shm,cuda_copy,cuda_ipc
    UCX_NET_DEVICES=mlx5_0:1
    UCX_IB_GPU_DIRECT_RDMA=yes
    UCX_MEMTYPE_CACHE=n
    UCX_UNIFIED_MODE=y
"
```

## Multi-Node Entrypoint Script

Create an advanced entrypoint script for multi-node coordination:

```bash
#!/bin/bash
# multi-node-entrypoint.sh

set -e

echo "=== Multi-Node Dask Cluster Setup ==="
echo "Node ID: $SLURM_NODEID"
echo "Local ID: $SLURM_LOCALID"
echo "Total Nodes: $SLURM_NNODES"
echo "Job ID: $SLURM_JOB_ID"

# === Network Configuration ===
detect_optimal_interface() {
    # Prefer InfiniBand, then high-speed Ethernet
    for pattern in "ib" "mlx" "25g" "10g" "eth"; do
        iface=$(ip link show | grep -i $pattern | head -1 | awk '{print $2}' | sed 's/://' || true)
        if [[ -n $iface ]] && [[ $iface != "lo" ]]; then
            echo $iface
            return
        fi
    done
    echo "eth0"  # fallback
}

if [[ -z "$INTERFACE" ]]; then
    export INTERFACE=$(detect_optimal_interface)
fi

echo "Using network interface: $INTERFACE"

# === Directory Setup ===
mkdir -p $LOGDIR $PROFILESDIR

# === Scheduler Setup (Node 0) ===
if [[ $SLURM_NODEID -eq 0 ]]; then
    echo "=== Starting Dask Scheduler on Node 0 ==="
    
    # Set scheduler-specific environment
    export DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES=10
    export DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL="300s"
    
    if [[ $DEVICE == "gpu" && $PROTOCOL == "ucx" ]]; then
        # GPU + UCX configuration
        DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True \
        DASK_DISTRIBUTED__RMM__POOL_SIZE=$RMM_SCHEDULER_POOL_SIZE \
        dask scheduler \
            --scheduler-file $SCHEDULER_FILE \
            --protocol $PROTOCOL \
            --interface $INTERFACE \
            --port 8786 \
            --dashboard-address 8787 >> $SCHEDULER_LOG 2>&1 &
    else
        # CPU or TCP configuration
        dask scheduler \
            --scheduler-file $SCHEDULER_FILE \
            --protocol $PROTOCOL \
            --interface $INTERFACE \
            --port 8786 \
            --dashboard-address 8787 >> $SCHEDULER_LOG 2>&1 &
    fi
    
    echo "Scheduler started, waiting for initialization..."
    sleep 45  # Extended wait for large clusters
fi

# === Wait for Scheduler ===
echo "Waiting for scheduler file..."
while [[ ! -f $SCHEDULER_FILE ]]; do
    sleep 5
done

echo "Scheduler file found, starting workers..."

# === Worker Setup (All Nodes) ===
export WORKER_LOG=$LOGDIR/worker_${SLURM_NODEID}.log

if [[ $DEVICE == "gpu" ]]; then
    # GPU Workers
    echo "Starting GPU workers on node $SLURM_NODEID"
    
    if [[ $PROTOCOL == "ucx" ]]; then
        # UCX GPU workers
        dask-cuda-worker \
            --scheduler-file $SCHEDULER_FILE \
            --rmm-pool-size $RMM_WORKER_POOL_SIZE \
            --interface $INTERFACE \
            --enable-cudf-spill \
            --rmm-async \
            --local-directory $JOB_DIR/worker-$SLURM_NODEID >> $WORKER_LOG 2>&1 &
    else
        # TCP GPU workers
        dask-cuda-worker \
            --scheduler-file $SCHEDULER_FILE \
            --rmm-pool-size $RMM_WORKER_POOL_SIZE \
            --interface $INTERFACE \
            --enable-cudf-spill \
            --rmm-async >> $WORKER_LOG 2>&1 &
    fi
else
    # CPU Workers
    echo "Starting CPU workers on node $SLURM_NODEID"
    dask worker \
        --scheduler-file $SCHEDULER_FILE \
        --memory-limit $CPU_WORKER_MEMORY_LIMIT \
        --nworkers $CPU_WORKERS_PER_NODE \
        --nthreads 2 \
        --interface $INTERFACE \
        --local-directory $JOB_DIR/worker-$SLURM_NODEID >> $WORKER_LOG 2>&1 &
fi

# === Extended Worker Startup Wait ===
echo "Workers starting, waiting for cluster stabilization..."
sleep 90  # Extended wait for large multi-node clusters

# === Cluster Health Check ===
if [[ $SLURM_NODEID -eq 0 ]]; then
    echo "=== Cluster Health Check ==="
    python -c "
import dask
from dask.distributed import Client
import time

try:
    client = Client(scheduler_file='$SCHEDULER_FILE', timeout='60s')
    print(f'Connected to scheduler: {client.scheduler.address}')
    print(f'Total workers: {len(client.scheduler_info()[\"workers\"])}')
    print(f'Total cores: {sum(w[\"nthreads\"] for w in client.scheduler_info()[\"workers\"].values())}')
    client.close()
    print('Cluster health check passed')
except Exception as e:
    print(f'Cluster health check failed: {e}')
    exit(1)
"
fi

# === Execute Main Script (Node 0 Only) ===
if [[ $SLURM_NODEID -eq 0 ]]; then
    echo "=== Executing Main Script ==="
    echo "Command: $SCRIPT_COMMAND"
    
    # Execute with error handling
    if bash -c "$SCRIPT_COMMAND"; then
        echo "Script completed successfully"
        touch $DONE_MARKER
    else
        echo "Script failed with exit code $?"
        touch $JOB_DIR/failed.marker
    fi
fi

# === Wait for Completion ===
echo "Waiting for job completion..."
while [[ ! -f $DONE_MARKER && ! -f $JOB_DIR/failed.marker ]]; do
    sleep 30
done

if [[ -f $JOB_DIR/failed.marker ]]; then
    echo "Job failed"
    exit 1
else
    echo "Job completed successfully"
fi

echo "Node $SLURM_NODEID finishing..."
```

## Monitoring and Troubleshooting

### Real-time Cluster Monitoring

```bash
# cluster_monitor.sh - Monitor multi-node cluster health
#!/bin/bash

monitor_cluster() {
    local scheduler_file=$1
    
    echo "=== Dask Cluster Monitor ==="
    python -c "
from dask.distributed import Client
import time
import json

client = Client(scheduler_file='$scheduler_file', timeout='30s')

while True:
    try:
        info = client.scheduler_info()
        workers = info['workers']
        
        print(f'\\nActive Workers: {len(workers)}')
        print(f'Total Cores: {sum(w[\"nthreads\"] for w in workers.values())}')
        
        # Memory usage
        total_memory = sum(w['memory_limit'] for w in workers.values()) / 1e9
        used_memory = sum(w['metrics']['memory'] for w in workers.values()) / 1e9
        print(f'Memory: {used_memory:.1f}GB / {total_memory:.1f}GB ({used_memory/total_memory*100:.1f}%)')
        
        # Task status
        tasks = client.scheduler.task_stream.buffer
        if tasks:
            print(f'Recent Tasks: {len(tasks)}')
        
        time.sleep(30)
        
    except Exception as e:
        print(f'Monitor error: {e}')
        break
"
}

# Usage
monitor_cluster $SCHEDULER_FILE
```

### Common Issues and Solutions

**Network connectivity problems:**
```bash
# Debug network connectivity
debug_network() {
    echo "=== Network Debugging ==="
    
    # Check interface status
    ip addr show $INTERFACE
    
    # Check connectivity between nodes
    for node in $(scontrol show hostname $SLURM_JOB_NODELIST); do
        echo "Testing connectivity to $node..."
        ping -c 3 $node || echo "WARNING: Cannot reach $node"
    done
    
    # Check port availability
    netstat -tuln | grep :8786 || echo "Scheduler port not open"
}
```

**Memory exhaustion handling:**
```bash
# Memory monitoring and cleanup
monitor_memory() {
    while true; do
        mem_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
        if (( $(echo "$mem_usage > 90" | bc -l) )); then
            echo "WARNING: Memory usage at ${mem_usage}%"
            # Trigger garbage collection in Dask workers
            python -c "
from dask.distributed import Client
client = Client(scheduler_file='$SCHEDULER_FILE')
client.run(lambda: __import__('gc').collect())
"
        fi
        sleep 60
    done
}
```

## Production Deployment Patterns

### Job Dependency Management

```bash
# Pipeline with job dependencies
#!/bin/bash

# Submit preprocessing job
PREPROCESS_JOB=$(sbatch --parsable preprocess_job.sh)

# Submit deduplication job (depends on preprocessing)
DEDUP_JOB=$(sbatch --parsable --dependency=afterok:$PREPROCESS_JOB dedup_job.sh)

# Submit final processing (depends on deduplication)
FINAL_JOB=$(sbatch --parsable --dependency=afterok:$DEDUP_JOB final_job.sh)

echo "Pipeline submitted:"
echo "  Preprocess: $PREPROCESS_JOB"
echo "  Dedup: $DEDUP_JOB" 
echo "  Final: $FINAL_JOB"
```

### Auto-scaling Configuration

```bash
# Auto-scaling based on queue depth
#!/bin/bash

monitor_and_scale() {
    while true; do
        # Check queue depth
        queue_depth=$(squeue -u $USER -h | wc -l)
        
        # Check cluster utilization
        active_nodes=$(sinfo -N -h -t idle | wc -l)
        
        if [[ $queue_depth -gt 10 && $active_nodes -lt 32 ]]; then
            echo "High queue depth, requesting more nodes..."
            sbatch --nodes=16 scale_up_job.sh
        fi
        
        sleep 300  # Check every 5 minutes
    done
}
```

## Performance Benchmarking

### Scaling Efficiency Testing

```bash
# Test scaling efficiency across different node counts
#!/bin/bash

test_scaling() {
    local dataset_size=$1
    
    for nodes in 2 4 8 16; do
        echo "Testing $nodes nodes with dataset size $dataset_size"
        
        start_time=$(date +%s)
        sbatch --wait --nodes=$nodes \
            --job-name="scale-test-${nodes}n" \
            --export=DATASET_SIZE=$dataset_size \
            scaling_test_job.sh
        end_time=$(date +%s)
        
        duration=$((end_time - start_time))
        throughput=$((dataset_size / duration))
        
        echo "$nodes,$dataset_size,$duration,$throughput" >> scaling_results.csv
    done
}

# Test with different dataset sizes
test_scaling 100GB
test_scaling 500GB
test_scaling 1TB
```

## References

- [Dask Distributed Best Practices](https://distributed.dask.org/en/latest/best-practices.html)
- [UCX Documentation](https://openucx.readthedocs.io/en/master/)
- [Slurm Multi-Node Configuration](https://slurm.schedmd.com/multi_cluster.html)
- [RAPIDS Multi-GPU Setup](https://docs.rapids.ai/api/dask-cuda/stable/) 