---
description: "Optimize ASR inference performance through efficient batch processing strategies and resource management"
categories: ["audio-processing"]
tags: ["batch-processing", "performance-optimization", "gpu-utilization", "memory-management", "scalability"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "audio-only"
---

# ASR Batch Processing Optimization

Optimize automatic speech recognition inference performance through efficient batch processing strategies, resource management, and scalability techniques for large-scale audio datasets.

## Batch Processing Fundamentals

### Why Batch Processing Matters

**GPU Utilization**: Batching maximizes GPU throughput by:
- Reducing model loading overhead
- Improving memory bandwidth utilization  
- Enabling parallel processing of multiple audio files
- Amortizing fixed costs across multiple samples

**Memory Efficiency**: Proper batching:
- Reduces memory fragmentation
- Enables efficient tensor operations
- Optimizes GPU memory allocation patterns
- Minimizes data transfer overhead

## Batch Size Optimization

### Hardware-Based Sizing

```python
def calculate_optimal_batch_size(gpu_memory_gb: float, model_size: str, 
                                audio_duration_avg: float) -> int:
    """Calculate optimal batch size based on hardware constraints."""
    
    # Base memory requirements (approximate)
    model_memory_gb = {
        "small": 2.0,
        "medium": 4.0, 
        "large": 8.0
    }
    
    # Memory per sample (varies with audio duration)
    memory_per_sample_mb = audio_duration_avg * 50  # Rough estimate
    
    # Available memory for batching
    available_memory_gb = gpu_memory_gb - model_memory_gb.get(model_size, 4.0)
    available_memory_mb = available_memory_gb * 1024
    
    # Calculate batch size with safety margin
    theoretical_batch_size = int(available_memory_mb / memory_per_sample_mb)
    optimal_batch_size = max(1, int(theoretical_batch_size * 0.8))  # 80% safety margin
    
    return min(optimal_batch_size, 128)  # Cap at reasonable maximum

# Usage examples
batch_sizes = {
    "rtx_3090_24gb": calculate_optimal_batch_size(24, "large", 5.0),    # ~32
    "rtx_4080_16gb": calculate_optimal_batch_size(16, "medium", 5.0),   # ~48  
    "rtx_4060_8gb": calculate_optimal_batch_size(8, "small", 5.0)       # ~64
}
```

### Dynamic Batch Sizing

```python
@dataclass
class DynamicBatchAsrStage(InferenceAsrNemoStage):
    """ASR stage with dynamic batch size adjustment."""
    
    initial_batch_size: int = 32
    min_batch_size: int = 4
    max_batch_size: int = 128
    
    def __post_init__(self):
        super().__post_init__()
        self.current_batch_size = self.initial_batch_size
        self.oom_count = 0
    
    def process(self, task: AudioBatch) -> AudioBatch:
        """Process with dynamic batch size adjustment."""
        
        while True:
            try:
                # Attempt processing with current batch size
                if len(task.data) > self.current_batch_size:
                    # Split large tasks into smaller batches
                    return self._process_in_batches(task)
                else:
                    return super().process(task)
                    
            except torch.cuda.OutOfMemoryError:
                self.oom_count += 1
                
                # Reduce batch size
                self.current_batch_size = max(
                    self.min_batch_size,
                    self.current_batch_size // 2
                )
                
                logger.warning(f"GPU OOM, reducing batch size to {self.current_batch_size}")
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                if self.current_batch_size < self.min_batch_size:
                    raise RuntimeError("Cannot reduce batch size further")
    
    def _process_in_batches(self, task: AudioBatch) -> AudioBatch:
        """Process large task in smaller batches."""
        
        all_results = []
        
        for i in range(0, len(task.data), self.current_batch_size):
            batch_data = task.data[i:i + self.current_batch_size]
            
            batch_task = AudioBatch(
                data=batch_data,
                filepath_key=task.filepath_key,
                task_id=f"{task.task_id}_batch_{i}"
            )
            
            batch_result = super().process(batch_task)
            all_results.extend(batch_result.data)
        
        return AudioBatch(
            data=all_results,
            filepath_key=task.filepath_key,
            task_id=task.task_id,
            dataset_name=task.dataset_name
        )
```

### Duration-Aware Batching

```python
def create_duration_balanced_batches(audio_files: list[dict], 
                                   target_batch_duration: float = 120.0) -> list[AudioBatch]:
    """Create batches with balanced total duration."""
    
    # Sort by duration for better packing
    sorted_files = sorted(audio_files, key=lambda x: x.get("duration", 0))
    
    batches = []
    current_batch = []
    current_duration = 0.0
    
    for audio_file in sorted_files:
        duration = audio_file.get("duration", 0)
        
        # Check if adding this file exceeds target duration
        if current_duration + duration > target_batch_duration and current_batch:
            # Create batch with current files
            batch = AudioBatch(
                data=current_batch,
                filepath_key="audio_filepath",
                task_id=f"duration_batch_{len(batches):04d}"
            )
            batches.append(batch)
            
            current_batch = []
            current_duration = 0.0
        
        current_batch.append(audio_file)
        current_duration += duration
    
    # Handle remaining files
    if current_batch:
        batch = AudioBatch(
            data=current_batch,
            filepath_key="audio_filepath",
            task_id=f"duration_batch_{len(batches):04d}"
        )
        batches.append(batch)
    
    return batches
```

## Memory Management

### GPU Memory Optimization

```python
import torch

def setup_memory_efficient_asr(model_name: str) -> InferenceAsrNemoStage:
    """Configure ASR stage for memory efficiency."""
    
    # Enable memory optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    asr_stage = InferenceAsrNemoStage(
        model_name=model_name
    ).with_(
        batch_size=16,  # Conservative batch size
        resources=Resources(gpus=1.0)
    )
    
    return asr_stage

def monitor_memory_usage(stage: InferenceAsrNemoStage, test_batch: AudioBatch):
    """Monitor GPU memory usage during processing."""
    
    if torch.cuda.is_available():
        # Clear cache before monitoring
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Process batch
        result = stage.process(test_batch)
        
        # Report memory statistics
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        print(f"Peak GPU memory: {peak_memory:.2f} GB")
        print(f"Current GPU memory: {current_memory:.2f} GB")
        print(f"Memory per sample: {peak_memory / len(test_batch.data):.3f} GB")
        
        return {
            "peak_memory_gb": peak_memory,
            "current_memory_gb": current_memory,
            "memory_per_sample_gb": peak_memory / len(test_batch.data)
        }
```

### Memory-Constrained Processing

```python
def create_memory_constrained_pipeline(max_gpu_memory_gb: float) -> Pipeline:
    """Create ASR pipeline optimized for limited GPU memory."""
    
    pipeline = Pipeline(name="memory_constrained_asr")
    
    # Use smaller model if memory is very limited
    if max_gpu_memory_gb < 8:
        model_name = "nvidia/stt_en_fastconformer_hybrid_small_pc"
        batch_size = 64
    elif max_gpu_memory_gb < 16:
        model_name = "nvidia/stt_en_fastconformer_hybrid_medium_pc"
        batch_size = 32
    else:
        model_name = "nvidia/stt_en_fastconformer_hybrid_large_pc"
        batch_size = 16
    
    # Configure ASR stage with memory limits
    asr_stage = InferenceAsrNemoStage(
        model_name=model_name
    ).with_(
        batch_size=batch_size,
        resources=Resources(
            gpus=min(1.0, max_gpu_memory_gb / 8),  # Scale GPU allocation
            memory=f"{max_gpu_memory_gb}GB"
        )
    )
    
    pipeline.add_stage(asr_stage)
    
    return pipeline
```

## Distributed Processing

### Multi-GPU Scaling

```python
def create_multi_gpu_asr_pipeline(model_name: str, num_gpus: int) -> Pipeline:
    """Scale ASR processing across multiple GPUs."""
    
    pipeline = Pipeline(name="multi_gpu_asr")
    
    # Data partitioning stage
    @processing_stage(name="gpu_partitioner")
    def partition_for_gpus(audio_batch: AudioBatch) -> list[AudioBatch]:
        """Partition audio batch across GPUs."""
        
        samples_per_gpu = len(audio_batch.data) // num_gpus
        partitions = []
        
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * samples_per_gpu
            
            if gpu_id == num_gpus - 1:
                # Last GPU gets remaining samples
                end_idx = len(audio_batch.data)
            else:
                end_idx = start_idx + samples_per_gpu
            
            partition_data = audio_batch.data[start_idx:end_idx]
            
            if partition_data:
                partition = AudioBatch(
                    data=partition_data,
                    filepath_key=audio_batch.filepath_key,
                    task_id=f"{audio_batch.task_id}_gpu_{gpu_id}",
                    dataset_name=f"{audio_batch.dataset_name}_gpu_{gpu_id}"
                )
                partitions.append(partition)
        
        return partitions
    
    pipeline.add_stage(partition_for_gpus)
    
    # Parallel ASR processing
    for gpu_id in range(num_gpus):
        asr_stage = InferenceAsrNemoStage(
            model_name=model_name
        ).with_(
            resources=Resources(gpus=1.0, device_id=gpu_id),
            batch_size=16
        )
        
        pipeline.add_parallel_stage(asr_stage, worker_id=gpu_id)
    
    # Merge results
    @processing_stage(name="result_merger")
    def merge_gpu_results(results: list[AudioBatch]) -> AudioBatch:
        """Merge results from multiple GPUs."""
        
        all_data = []
        for result_batch in results:
            all_data.extend(result_batch.data)
        
        return AudioBatch(
            data=all_data,
            filepath_key=results[0].filepath_key,
            task_id="merged_gpu_results",
            dataset_name=results[0].dataset_name.split("_gpu_")[0]
        )
    
    pipeline.add_stage(merge_gpu_results)
    
    return pipeline
```

### Cluster Scaling

```python
from nemo_curator.backends.xenna import XennaExecutor

def create_cluster_asr_executor(num_workers: int, gpus_per_worker: int = 1) -> XennaExecutor:
    """Create executor for cluster-scale ASR processing."""
    
    executor = XennaExecutor(
        num_workers=num_workers,
        resources_per_worker=Resources(
            gpus=gpus_per_worker,
            cpus=4,
            memory="16GB"
        ),
        
        # Cluster-specific configuration
        worker_startup_timeout=300,  # Allow time for model loading
        max_retries=3,              # Retry failed tasks
        
        # Ray configuration
        ray_init_kwargs={
            "num_gpus": num_workers * gpus_per_worker,
            "object_store_memory": 10**9 * 4  # 4GB object store
        }
    )
    
    return executor

# Usage for large-scale processing
cluster_executor = create_cluster_asr_executor(
    num_workers=8,      # 8 worker nodes
    gpus_per_worker=2   # 2 GPUs per worker = 16 total GPUs
)
```

## Performance Monitoring

### Throughput Tracking

```python
class PerformanceMonitor:
    """Monitor ASR processing performance."""
    
    def __init__(self):
        self.start_time = None
        self.total_samples = 0
        self.total_audio_duration = 0.0
        self.batch_times = []
    
    def start_monitoring(self):
        self.start_time = time.time()
    
    def record_batch(self, batch_size: int, audio_duration: float, processing_time: float):
        self.total_samples += batch_size
        self.total_audio_duration += audio_duration
        self.batch_times.append(processing_time)
    
    def get_statistics(self) -> dict:
        if not self.start_time:
            return {}
        
        total_time = time.time() - self.start_time
        
        return {
            "total_processing_time": total_time,
            "total_samples": self.total_samples,
            "total_audio_hours": self.total_audio_duration / 3600,
            "samples_per_second": self.total_samples / total_time,
            "real_time_factor": self.total_audio_duration / total_time,
            "average_batch_time": np.mean(self.batch_times),
            "batch_time_std": np.std(self.batch_times),
            "throughput_audio_hours_per_hour": (self.total_audio_duration / 3600) / (total_time / 3600)
        }

# Usage in pipeline
monitor = PerformanceMonitor()
monitor.start_monitoring()

# Track each batch
for batch in audio_batches:
    batch_start = time.time()
    result = asr_stage.process(batch)
    batch_time = time.time() - batch_start
    
    batch_duration = sum(item["duration"] for item in batch.data)
    monitor.record_batch(len(batch.data), batch_duration, batch_time)

# Get final statistics
stats = monitor.get_statistics()
print(f"Processed {stats['total_samples']} samples at {stats['samples_per_second']:.2f} samples/sec")
```

### Adaptive Batch Sizing

```python
class AdaptiveBatchProcessor:
    """Automatically adjust batch size based on performance."""
    
    def __init__(self, asr_stage: InferenceAsrNemoStage, target_memory_usage: float = 0.8):
        self.asr_stage = asr_stage
        self.target_memory_usage = target_memory_usage
        self.current_batch_size = asr_stage.batch_size
        self.performance_history = []
    
    def process_adaptive_batch(self, audio_data: list[dict]) -> list[dict]:
        """Process audio data with adaptive batch sizing."""
        
        results = []
        
        for i in range(0, len(audio_data), self.current_batch_size):
            batch_data = audio_data[i:i + self.current_batch_size]
            
            batch = AudioBatch(
                data=batch_data,
                filepath_key="audio_filepath"
            )
            
            try:
                # Monitor memory before processing
                initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                # Process batch
                start_time = time.time()
                result = self.asr_stage.process(batch)
                processing_time = time.time() - start_time
                
                # Monitor memory after processing
                peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                
                # Record performance
                self.performance_history.append({
                    "batch_size": len(batch_data),
                    "processing_time": processing_time,
                    "peak_memory": peak_memory,
                    "samples_per_second": len(batch_data) / processing_time
                })
                
                results.extend(result.data)
                
                # Adjust batch size based on memory usage
                self._adjust_batch_size(peak_memory)
                
            except torch.cuda.OutOfMemoryError:
                # Reduce batch size and retry
                self.current_batch_size = max(1, self.current_batch_size // 2)
                torch.cuda.empty_cache()
                
                # Retry with smaller batch
                smaller_batch = AudioBatch(data=batch_data[:self.current_batch_size])
                result = self.asr_stage.process(smaller_batch)
                results.extend(result.data)
        
        return results
    
    def _adjust_batch_size(self, peak_memory: int):
        """Adjust batch size based on memory usage."""
        
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_usage_ratio = peak_memory / total_memory
            
            if memory_usage_ratio < self.target_memory_usage * 0.7:
                # Increase batch size if memory usage is low
                self.current_batch_size = min(
                    self.current_batch_size * 2,
                    128  # Maximum batch size
                )
            elif memory_usage_ratio > self.target_memory_usage:
                # Decrease batch size if memory usage is high
                self.current_batch_size = max(
                    self.current_batch_size // 2,
                    1  # Minimum batch size
                )
```

## Parallel Processing Strategies

### Thread-Based Parallelism

```python
from concurrent.futures import ThreadPoolExecutor
import queue

def parallel_batch_processing(audio_batches: list[AudioBatch], 
                            asr_stage: InferenceAsrNemoStage,
                            max_workers: int = 4) -> list[AudioBatch]:
    """Process multiple batches in parallel using threads."""
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches for processing
        future_to_batch = {
            executor.submit(asr_stage.process, batch): batch 
            for batch in audio_batches
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Optionally retry or skip
    
    return results
```

### Pipeline Parallelism

```python
def create_pipelined_asr_processing() -> Pipeline:
    """Create pipeline with overlapped processing stages."""
    
    pipeline = Pipeline(name="pipelined_asr")
    
    # Stage 1: Data loading (CPU)
    pipeline.add_stage(
        data_loading_stage.with_(
            resources=Resources(cpus=2.0),
            prefetch_batches=3  # Load ahead of processing
        )
    )
    
    # Stage 2: ASR inference (GPU)
    pipeline.add_stage(
        asr_inference_stage.with_(
            resources=Resources(gpus=1.0),
            pipeline_parallel=True  # Enable pipeline parallelism
        )
    )
    
    # Stage 3: Post-processing (CPU)
    pipeline.add_stage(
        quality_assessment_stage.with_(
            resources=Resources(cpus=2.0),
            async_processing=True  # Process while next batch loads
        )
    )
    
    return pipeline
```

## Batch Processing Patterns

### Streaming Processing

```python
def streaming_asr_processor(audio_stream: Iterator[AudioBatch], 
                          asr_stage: InferenceAsrNemoStage) -> Iterator[AudioBatch]:
    """Process audio batches in streaming fashion."""
    
    for batch in audio_stream:
        try:
            # Process batch
            result = asr_stage.process(batch)
            yield result
            
            # Optional: Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Streaming processing error: {e}")
            # Yield empty result or skip batch
            yield AudioBatch(data=[], filepath_key=batch.filepath_key)
```

### Checkpointed Processing

```python
def checkpointed_batch_processing(audio_batches: list[AudioBatch],
                                asr_stage: InferenceAsrNemoStage,
                                checkpoint_dir: str) -> list[AudioBatch]:
    """Process batches with checkpointing for recovery."""
    
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    
    results = []
    
    for i, batch in enumerate(audio_batches):
        checkpoint_file = checkpoint_path / f"batch_{i:04d}.jsonl"
        
        # Check if batch already processed
        if checkpoint_file.exists():
            logger.info(f"Loading cached result for batch {i}")
            
            with open(checkpoint_file, 'r') as f:
                cached_data = [json.loads(line) for line in f]
            
            cached_batch = AudioBatch(
                data=cached_data,
                filepath_key=batch.filepath_key,
                task_id=batch.task_id
            )
            results.append(cached_batch)
            
        else:
            # Process batch
            logger.info(f"Processing batch {i}/{len(audio_batches)}")
            
            try:
                result = asr_stage.process(batch)
                results.append(result)
                
                # Save checkpoint
                with open(checkpoint_file, 'w') as f:
                    for item in result.data:
                        f.write(json.dumps(item) + '\n')
                        
            except Exception as e:
                logger.error(f"Batch {i} failed: {e}")
                # Save error state
                error_file = checkpoint_path / f"batch_{i:04d}_error.txt"
                with open(error_file, 'w') as f:
                    f.write(str(e))
    
    return results
```

## Performance Benchmarking

### Batch Size Benchmarking

```python
def benchmark_batch_sizes(audio_data: list[dict], model_name: str, 
                        batch_sizes: list[int]) -> dict:
    """Benchmark different batch sizes for optimal performance."""
    
    results = {}
    
    asr_stage = InferenceAsrNemoStage(model_name=model_name)
    asr_stage.setup()
    
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size: {batch_size}")
        
        # Create test batches
        test_batches = create_size_based_batches(audio_data[:100], batch_size)
        
        # Measure performance
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        for batch in test_batches:
            asr_stage.process(batch)
        
        processing_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        results[batch_size] = {
            "processing_time": processing_time,
            "samples_per_second": 100 / processing_time,
            "peak_memory_gb": peak_memory,
            "memory_per_sample": peak_memory / 100
        }
        
        # Clear cache between tests
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

# Find optimal batch size
benchmark_results = benchmark_batch_sizes(
    audio_data=sample_audio_files,
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
    batch_sizes=[4, 8, 16, 32, 64]
)

# Select batch size with best samples/second ratio
optimal_batch_size = max(
    benchmark_results.keys(),
    key=lambda bs: benchmark_results[bs]["samples_per_second"]
)
```
