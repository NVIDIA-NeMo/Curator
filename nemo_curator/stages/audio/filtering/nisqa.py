# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NISQA (Non-Intrusive Speech Quality Assessment) filter stage.

Filters audio segments based on NISQA quality metrics including
MOS, noisiness, coloration, discontinuity, and loudness.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.filtering import NISQAFilterStage
    from nemo_curator.stages.resources import Resources
    
    pipeline = Pipeline(name="quality_pipeline")
    pipeline.add_stage(
        NISQAFilterStage(mos_threshold=4.5, noi_threshold=4.3)
        .with_(resources=Resources(gpus=0.3))
    )
"""

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

import torch
from loguru import logger
from pydub import AudioSegment

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from ..configs import NISQAConfig


def _nisqa_worker_init(gpu_id: int):
    """Initialize worker with specific GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)


# Thread-local storage for per-thread NISQA pipeline instances
import threading
_NISQA_THREAD_LOCAL = threading.local()


def _get_nisqa_pipeline_for_thread(gpu_id: int, model_path: str):
    """Get or create a NISQAPipeline for the current thread and GPU."""
    from nemo_curator.stages.audio.filtering.nisqa_filter_module.nisqa_pipeline import NISQAPipeline
    
    # Use thread-local storage to avoid sharing models between threads
    cache_key = f"nisqa_gpu_{gpu_id}"
    
    if not hasattr(_NISQA_THREAD_LOCAL, 'pipelines'):
        _NISQA_THREAD_LOCAL.pipelines = {}
    
    if cache_key not in _NISQA_THREAD_LOCAL.pipelines:
        # Set GPU before creating pipeline
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        
        config = {'nisqa': {'model_path': model_path}}
        _NISQA_THREAD_LOCAL.pipelines[cache_key] = NISQAPipeline(
            gpu_id=gpu_id, 
            config=config
        )
    
    return _NISQA_THREAD_LOCAL.pipelines[cache_key]


def _nisqa_process_on_gpu(
    item_data: Tuple[Dict[str, Any], str, str, dict, dict],
    gpu_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Process a single item on a specific GPU.
    
    This function is designed to be called in a separate thread
    with a specific GPU assignment. Uses thread-local model instances
    to avoid thread-safety issues.
    """
    item, task_id, model_path, thresholds, config = item_data
    
    try:
        # Set GPU for this thread
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        
        audio = item.get('audio')
        if audio is None:
            audio_filepath = item.get('audio_filepath')
            if audio_filepath and os.path.exists(audio_filepath):
                audio = AudioSegment.from_wav(audio_filepath)
                item['audio'] = audio
            else:
                return None
        
        # Get thread-local pipeline for this GPU
        pipeline = _get_nisqa_pipeline_for_thread(gpu_id, model_path)
        
        temp_path = None
        try:
            # Use unique temp file with thread ID to avoid conflicts
            thread_id = threading.current_thread().ident
            with tempfile.NamedTemporaryFile(
                suffix=f'_gpu{gpu_id}_t{thread_id}.wav', 
                delete=False
            ) as temp_file:
                temp_path = temp_file.name
            
            audio.export(temp_path, format="wav")
            
            # CRITICAL: Always set GPU device before prediction
            # Thread may have handled a different GPU since pipeline was cached
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
            
            # Use the thread-local pipeline directly
            score_data = pipeline.predict_file(temp_path)
            
            if isinstance(score_data, dict):
                mos = float(score_data.get('mos_pred', 0))
                noi = float(score_data.get('noi_pred', 0))
                col = float(score_data.get('col_pred', 0))
                dis = float(score_data.get('dis_pred', 0))
                loud = float(score_data.get('loud_pred', 0))
            else:
                mos = noi = col = dis = loud = float(score_data)
            
            # Debug: log the actual scores with segment info
            segment_num = item.get('segment_num', 'unknown')
            if mos < 0.1:  # Suspiciously low score
                logger.warning(f"[GPU {gpu_id}] Segment {segment_num}: LOW MOS={mos:.4f}, score_data={score_data}")
            
            # Check thresholds
            passed = True
            if thresholds.get('mos') is not None and mos < thresholds['mos']:
                passed = False
            if thresholds.get('noi') is not None and noi < thresholds['noi']:
                passed = False
            if thresholds.get('col') is not None and col < thresholds['col']:
                passed = False
            if thresholds.get('dis') is not None and dis < thresholds['dis']:
                passed = False
            if thresholds.get('loud') is not None and loud < thresholds['loud']:
                passed = False
            
            if passed:
                item['nisqa_mos'] = mos
                item['nisqa_noi'] = noi
                item['nisqa_col'] = col
                item['nisqa_dis'] = dis
                item['nisqa_loud'] = loud
                return item
            else:
                return None
                
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"[{task_id}] Error in multi-GPU NISQA processing on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _load_audio_as_pydub(audio_path: str) -> AudioSegment:
    """
    Load audio file as PyDub AudioSegment.
    
    Supports standalone usage of stages without requiring previous stages.
    """
    return AudioSegment.from_wav(audio_path)


@dataclass
class NISQAFilterStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    NISQA quality assessment filter stage.
    
    Filters audio segments based on NISQA quality metrics.
    Returns None for segments that don't meet threshold requirements.
    
    NISQA predicts (1-5 scale):
    - MOS: Overall Mean Opinion Score
    - NOI: Noisiness (higher = less noisy)
    - COL: Coloration/distortion
    - DIS: Discontinuity
    - LOUD: Loudness appropriateness
    
    Args:
        config: NISQAConfig object (overrides other params if provided)
        model_path: Path to NISQA model weights
        mos_threshold: Minimum MOS score (None to disable)
        noi_threshold: Minimum noisiness score (None to disable)
        col_threshold: Minimum coloration score (None to disable)
        dis_threshold: Minimum discontinuity score (None to disable)
        loud_threshold: Minimum loudness score (None to disable)
    
    Note:
        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    
    Example:
        # Using config
        config = NISQAConfig(mos_threshold=4.5)
        stage = NISQAFilterStage(config=config)
        
        # Using parameters
        stage = NISQAFilterStage(mos_threshold=4.0, noi_threshold=4.0)
    """
    
    config: Optional[NISQAConfig] = None
    model_path: str = "model/nisqa.tar"
    mos_threshold: Optional[float] = 4.5
    noi_threshold: Optional[float] = 4.3
    col_threshold: Optional[float] = None
    dis_threshold: Optional[float] = None
    loud_threshold: Optional[float] = None
    
    name: str = "NISQAFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(gpus=0.3))
    
    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__()
        self._predict_function = None
        self._init_lock = None  # Lazy initialization to avoid pickle issues
        
        # Apply config if provided
        if self.config is not None:
            self.model_path = self.config.model_path
            self.mos_threshold = self.config.mos_threshold
            self.noi_threshold = self.config.noi_threshold
            self.col_threshold = self.config.col_threshold
            self.dis_threshold = self.config.dis_threshold
            self.loud_threshold = self.config.loud_threshold
            # Apply resources from config
            self.resources = Resources(cpus=self.config.cpus, gpus=self.config.gpus)
    
    def __getstate__(self):
        """Return state for pickling, excluding unpicklable objects."""
        state = self.__dict__.copy()
        # Remove the lock and predict function - they'll be recreated
        state['_init_lock'] = None
        state['_predict_function'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._init_lock = None
        self._predict_function = None
    
    def _get_lock(self):
        """Get or create the initialization lock (lazy initialization)."""
        if self._init_lock is None:
            self._init_lock = threading.Lock()
        return self._init_lock
    
    def inputs(self) -> Tuple[List[str], List[str]]:
        """Define required inputs."""
        return [], ["audio"]
    
    def outputs(self) -> Tuple[List[str], List[str]]:
        """Define outputs produced by this stage."""
        return [], ["nisqa_mos", "nisqa_noi", "nisqa_col", "nisqa_dis", "nisqa_loud"]
    
    def setup(self, worker_metadata=None) -> None:
        """Load NISQA model on worker initialization."""
        self._initialize_model()
    
    def teardown(self) -> None:
        """Clean up resources."""
        if self._predict_function is not None:
            self._predict_function = None
            import torch
            torch.cuda.empty_cache()
    
    def _initialize_model(self):
        """Initialize the NISQA prediction function (thread-safe)."""
        if self._predict_function is None:
            with self._get_lock():
                if self._predict_function is None:
                    try:
                        # Import from local nisqa_filter_module (same directory)
                        from nemo_curator.stages.audio.filtering.nisqa_filter_module.nisqa_pipeline import predict_batch_mos
                        self._predict_function = predict_batch_mos
                        logger.info("NISQA prediction function loaded successfully")
                    except ImportError as e:
                        logger.error(f"Failed to import NISQA module: {e}")
                        raise
    
    def _resolve_model_path(self) -> str:
        """Resolve model path to absolute path."""
        if os.path.isabs(self.model_path):
            return self.model_path
        
        # Try relative to nisqa_filter_module first (default location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        module_dir = os.path.join(current_dir, 'nisqa_filter_module')
        resolved = os.path.join(module_dir, self.model_path)
        if os.path.exists(resolved):
            return resolved
        
        # Try relative to filtering directory
        resolved = os.path.join(current_dir, self.model_path)
        if os.path.exists(resolved):
            return resolved
        
        # Return the module path as default
        return os.path.join(module_dir, self.model_path)
    
    def _process_single_item(self, item: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
        """Process a single audio item and return it with NISQA scores if it passes thresholds."""
        audio = item.get('audio')
        
        # Auto-load from file if audio not provided (standalone usage)
        if audio is None:
            audio_filepath = item.get('audio_filepath')
            if audio_filepath and os.path.exists(audio_filepath):
                try:
                    audio = _load_audio_as_pydub(audio_filepath)
                    item['audio'] = audio
                except Exception as e:
                    logger.error(f"[{task_id}] Failed to load audio file: {e}")
                    return None
            else:
                logger.warning(f"[{task_id}] No audio or valid audio_filepath found")
                return None
        
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            audio.export(temp_path, format="wav")
            
            model_path = self._resolve_model_path()
            pipeline_config = {'nisqa': {'model_path': model_path}}
            
            # Use current CUDA device (set by executor)
            if torch.cuda.is_available():
                selected_gpu = torch.cuda.current_device()
            else:
                selected_gpu = 0
            
            scores = self._predict_function(
                [temp_path], gpu_id=selected_gpu, config=pipeline_config
            )
            
            if temp_path not in scores:
                logger.warning(f"[{task_id}] No NISQA score returned")
                return None
            
            score_data = scores[temp_path]
            
            if isinstance(score_data, dict):
                mos = float(score_data.get('mos_pred', 0))
                noi = float(score_data.get('noi_pred', 0))
                col = float(score_data.get('col_pred', 0))
                dis = float(score_data.get('dis_pred', 0))
                loud = float(score_data.get('loud_pred', 0))
            else:
                mos = noi = col = dis = loud = float(score_data)
            
            # Log the actual scores for debugging
            logger.debug(
                f"[{task_id}] NISQA scores: MOS={mos:.3f}, NOI={noi:.3f}, COL={col:.3f}, "
                f"DIS={dis:.3f}, LOUD={loud:.3f}"
            )
            
            # Check thresholds
            passed = True
            fail_reasons = []
            if self.mos_threshold is not None and mos < self.mos_threshold:
                passed = False
                fail_reasons.append(f"MOS {mos:.3f} < {self.mos_threshold}")
            if self.noi_threshold is not None and noi < self.noi_threshold:
                passed = False
                fail_reasons.append(f"NOI {noi:.3f} < {self.noi_threshold}")
            if self.col_threshold is not None and col < self.col_threshold:
                passed = False
                fail_reasons.append(f"COL {col:.3f} < {self.col_threshold}")
            if self.dis_threshold is not None and dis < self.dis_threshold:
                passed = False
                fail_reasons.append(f"DIS {dis:.3f} < {self.dis_threshold}")
            if self.loud_threshold is not None and loud < self.loud_threshold:
                passed = False
                fail_reasons.append(f"LOUD {loud:.3f} < {self.loud_threshold}")
            
            if not passed:
                logger.info(f"[{task_id}] NISQA FAILED: {', '.join(fail_reasons)}")
            
            if passed:
                item['nisqa_mos'] = mos
                item['nisqa_noi'] = noi
                item['nisqa_col'] = col
                item['nisqa_dis'] = dis
                item['nisqa_loud'] = loud
                return item
            else:
                return None
                
        except Exception as e:
            logger.error(f"[{task_id}] Error in NISQA filtering: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _process_multi_gpu(self, task: AudioBatch, num_gpus: int) -> List[Dict[str, Any]]:
        """
        Process items in parallel across multiple GPUs.
        
        Args:
            task: AudioBatch with multiple items
            num_gpus: Number of GPUs to use
            
        Returns:
            List of processed items that passed thresholds
        """
        total_items = len(task.data)
        model_path = self._resolve_model_path()
        
        # Prepare thresholds dict
        thresholds = {
            'mos': self.mos_threshold,
            'noi': self.noi_threshold,
            'col': self.col_threshold,
            'dis': self.dis_threshold,
            'loud': self.loud_threshold,
        }
        
        # Prepare item data tuples for the worker function
        item_data_list = [
            (item, task.task_id, model_path, thresholds, {})
            for item in task.data
        ]
        
        # Get available GPU IDs
        available_gpus = list(range(num_gpus))
        
        results = []
        
        # Use ThreadPoolExecutor with GPU round-robin assignment
        # Each thread will set its own CUDA device
        max_workers = min(num_gpus, total_items)
        
        logger.info(f"[NISQAFilter] Using multi-GPU parallel processing: {max_workers} GPUs for {total_items} items")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, item_data in enumerate(item_data_list):
                gpu_id = available_gpus[i % num_gpus]  # Round-robin GPU assignment
                future = executor.submit(_nisqa_process_on_gpu, item_data, gpu_id)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"[{task.task_id}] Error in multi-GPU NISQA processing: {e}")
        
        return results

    def process(self, task: AudioBatch) -> Optional[AudioBatch]:
        """
        Filter audio based on NISQA quality scores.
        
        Automatically uses parallel processing when:
        - Multiple items in task.data (> 1)
        - Resources specify cpus > 1 (CPU parallelism via ThreadPoolExecutor)
        - Resources specify gpus > 1 (Multi-GPU parallelism)
        - Resources specify gpus > 0 and gpus <= 1 (Single GPU sequential)
        
        The stage discovers its allocated resources via self._resources
        (set via .with_(resources=Resources(...))) and optimizes accordingly.
        
        Args:
            task: AudioBatch with audio data
            
        Returns:
            AudioBatch with NISQA scores if passes, None otherwise
        """
        self._initialize_model()
        
        if self._predict_function is None:
            logger.error("NISQA prediction function not available")
            return AudioBatch(data=[], task_id=task.task_id, dataset_name=task.dataset_name)
        
        total_items = len(task.data)
        
        # Determine processing strategy based on resources and item count
        num_gpus = int(self._resources.gpus) if self._resources.gpus >= 1 else (1 if self._resources.gpus > 0 else 0)
        
        # Multi-GPU parallel processing
        use_multi_gpu = (
            total_items > 1 and
            num_gpus > 1 and
            torch.cuda.is_available() and
            torch.cuda.device_count() >= num_gpus
        )
        
        # CPU parallel processing (only when no GPU)
        use_cpu_parallel = (
            total_items > 1 and 
            self._resources.cpus > 1 and 
            self._resources.gpus == 0
        )
        
        if use_multi_gpu:
            # Multi-GPU parallel processing
            results = self._process_multi_gpu(task, num_gpus)
            mode = f"multi-GPU ({num_gpus} GPUs)"
        elif use_cpu_parallel:
            # CPU parallel processing using ThreadPoolExecutor
            max_workers = min(int(self._resources.cpus), total_items)
            logger.debug(f"[NISQAFilter] Using CPU parallel processing with {max_workers} workers for {total_items} items")
            
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._process_single_item, item, task.task_id)
                    for item in task.data
                ]
                for future in futures:
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"[{task.task_id}] Error in parallel NISQA processing: {e}")
            mode = "CPU parallel"
        else:
            # Sequential processing (default, or single GPU)
            results = []
            for item in task.data:
                result = self._process_single_item(item, task.task_id)
                if result is not None:
                    results.append(result)
            mode = "GPU" if self._resources.gpus > 0 else "sequential"
        
        passed_count = len(results)
        rejected_count = total_items - passed_count
        
        # Log summary
        threshold_parts = []
        if self.mos_threshold is not None:
            threshold_parts.append(f"MOS>={self.mos_threshold}")
        if self.noi_threshold is not None:
            threshold_parts.append(f"NOI>={self.noi_threshold}")
        if self.col_threshold is not None:
            threshold_parts.append(f"COL>={self.col_threshold}")
        if self.dis_threshold is not None:
            threshold_parts.append(f"DIS>={self.dis_threshold}")
        if self.loud_threshold is not None:
            threshold_parts.append(f"LOUD>={self.loud_threshold}")
        threshold_str = ", ".join(threshold_parts) if threshold_parts else "none"
        
        logger.info(f"[NISQAFilter] {task.task_id}: {passed_count}/{total_items} passed (thresholds: {threshold_str}) [{mode}]")
        
        return AudioBatch(data=results, task_id=task.task_id, dataset_name=task.dataset_name)
