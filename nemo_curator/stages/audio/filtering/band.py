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
Band filter stage for audio bandwidth classification.

Classifies audio as "full_band" or "narrow_band" based on spectral
characteristics. Useful for filtering low-quality telephone or compressed audio.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.filtering import BandFilterStage
    
    # Pass only full-band audio
    pipeline = Pipeline(name="band_pipeline")
    pipeline.add_stage(BandFilterStage(band_value="full_band"))
    
    # Pass only narrow-band audio
    pipeline.add_stage(BandFilterStage(band_value="narrow_band"))
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import soundfile as sf
import torch
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from ..configs import BandFilterConfig


def _band_process_on_gpu(
    item_data: Tuple[Dict[str, Any], str, str, str, str, int, int],
    gpu_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Process a single item on a specific GPU for feature extraction.
    
    This function is designed to be called in a separate thread
    with a specific GPU assignment for feature extraction parallelization.
    """
    item, task_id, model_path, feature_group, band_value, n_workers, cache_size = item_data
    
    try:
        # Set GPU for this thread (for feature extraction)
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        
        # Import predictor
        from nemo_curator.stages.audio.filtering.band_filter_module.predict import BandPredictor
        
        waveform = item.get('waveform')
        sample_rate = item.get('sample_rate', 48000)
        
        # Auto-load from file if waveform not provided
        if waveform is None:
            audio_filepath = item.get('audio_filepath')
            if audio_filepath and os.path.exists(audio_filepath):
                data, sample_rate = sf.read(audio_filepath, dtype='float32')
                waveform = torch.from_numpy(data)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                item['waveform'] = waveform
                item['sample_rate'] = sample_rate
            else:
                return None
        
        # Create predictor for this GPU
        predictor = BandPredictor(
            model_path=model_path,
            feature_group=feature_group,
            n_workers=1,  # Single worker within GPU thread
            feature_cache_size=cache_size,
            use_gpu=True,
        )
        
        # Predict
        audio_data = [(waveform, sample_rate)]
        predictions = predictor.predict_audio_batch(audio_data, use_parallel=False)
        
        if predictions and len(predictions) > 0:
            prediction = predictions[0]
            
            if isinstance(prediction, str) and prediction.startswith("Error"):
                return None
            
            # Check if prediction matches the desired band_value
            is_pass = prediction == band_value
            
            if is_pass:
                item['band_prediction'] = prediction
                return item
            else:
                return None
        else:
            return None
            
    except Exception as e:
        logger.error(f"[{task_id}] Error in multi-GPU Band processing on GPU {gpu_id}: {e}")
        return None


def _load_audio_file(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and return waveform tensor and sample rate.
    
    Supports standalone usage of stages without requiring MonoConversionStage.
    """
    data, sample_rate = sf.read(audio_path, dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


@dataclass
class BandFilterStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Band filter stage for bandwidth classification.
    
    Classifies audio as "full_band" or "narrow_band" and filters
    based on the specified band_value to pass.
    
    Args:
        config: BandFilterConfig object (overrides other params if provided)
        model_path: Path to band classifier model (.joblib)
        feature_group: Feature extraction group
        n_workers: Number of parallel workers
        feature_cache_size: Size of feature cache
        band_value: Which band type to pass ("full_band" or "narrow_band")
    
    Note:
        GPU is used automatically when resources specify gpus > 0.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    
    Example:
        # Using config - pass only full-band audio
        config = BandFilterConfig(band_value="full_band")
        stage = BandFilterStage(config=config)
        
        # Using parameters - pass only narrow-band audio
        stage = BandFilterStage(band_value="narrow_band")
    """
    
    config: Optional[BandFilterConfig] = None
    model_path: str = "model/band_classifier_model_band_7000_samples.joblib"
    feature_group: str = "band"
    n_workers: int = 4
    feature_cache_size: int = 100
    band_value: str = "full_band"
    
    name: str = "BandFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    
    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__()
        self._predictor = None
        self._init_lock = None  # Lazy initialization to avoid pickle issues
        
        # Apply config if provided
        if self.config is not None:
            self.model_path = self.config.model_path
            self.feature_group = self.config.feature_group
            self.n_workers = self.config.n_workers
            self.feature_cache_size = self.config.feature_cache_size
            self.band_value = self.config.band_value
            # Apply resources from config
            self.resources = Resources(cpus=self.config.cpus, gpus=self.config.gpus)
    
    def __getstate__(self):
        """Return state for pickling, excluding unpicklable objects."""
        state = self.__dict__.copy()
        # Remove the lock and predictor - they'll be recreated
        state['_init_lock'] = None
        state['_predictor'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._init_lock = None
        self._predictor = None
    
    def _get_lock(self):
        """Get or create the initialization lock (lazy initialization)."""
        if self._init_lock is None:
            self._init_lock = threading.Lock()
        return self._init_lock
    
    def inputs(self) -> Tuple[List[str], List[str]]:
        """Define required inputs."""
        return [], ["waveform", "sample_rate"]
    
    def outputs(self) -> Tuple[List[str], List[str]]:
        """Define outputs produced by this stage."""
        return [], ["band_prediction"]
    
    def setup(self, worker_metadata=None) -> None:
        """Load band predictor on worker initialization."""
        self._initialize_predictor()
    
    def teardown(self) -> None:
        """Clean up resources."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
            torch.cuda.empty_cache()
    
    def _initialize_predictor(self):
        """Initialize the band predictor (thread-safe)."""
        if self._predictor is None:
            with self._get_lock():
                if self._predictor is None:
                    try:
                        # Import from local band_filter_module (same directory)
                        from nemo_curator.stages.audio.filtering.band_filter_module.predict import BandPredictor
                        
                        model_path = self._resolve_model_path()
                        
                        # Determine GPU usage based on resources
                        use_gpu = self._resources.gpus > 0 and torch.cuda.is_available()
                        
                        self._predictor = BandPredictor(
                            model_path=model_path,
                            feature_group=self.feature_group,
                            n_workers=self.n_workers,
                            feature_cache_size=self.feature_cache_size,
                            use_gpu=use_gpu,
                        )
                        logger.info(f"Band predictor loaded successfully (GPU: {use_gpu})")
                    except ImportError as e:
                        logger.error(f"Failed to import Band module: {e}")
                        raise
    
    def _resolve_model_path(self) -> str:
        """Resolve model path to absolute path."""
        if os.path.isabs(self.model_path):
            return self.model_path
        
        # Try relative to band_filter_module first (default location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        module_dir = os.path.join(current_dir, 'band_filter_module')
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
        """Process a single audio item and return it with band prediction if it passes."""
        waveform = item.get('waveform')
        sample_rate = item.get('sample_rate', 48000)
        
        # Auto-load from file if waveform not provided (standalone usage)
        if waveform is None:
            audio_filepath = item.get('audio_filepath')
            if audio_filepath and os.path.exists(audio_filepath):
                try:
                    waveform, sample_rate = _load_audio_file(audio_filepath)
                    item['waveform'] = waveform
                    item['sample_rate'] = sample_rate
                except Exception as e:
                    logger.error(f"[{task_id}] Failed to load audio file: {e}")
                    return None
            else:
                logger.warning(f"[{task_id}] No waveform or valid audio_filepath found")
                return None
        
        try:
            audio_data = [(waveform, sample_rate)]
            predictions = self._predictor.predict_audio_batch(audio_data, use_parallel=False)
            
            if predictions and len(predictions) > 0:
                prediction = predictions[0]
                
                if isinstance(prediction, str) and prediction.startswith("Error"):
                    logger.error(f"[{task_id}] Band prediction error: {prediction}")
                    return None
                
                # Check if prediction matches the desired band_value
                is_pass = prediction == self.band_value
                
                if is_pass:
                    item['band_prediction'] = prediction
                    return item
                else:
                    return None
            else:
                logger.warning(f"[{task_id}] No prediction returned from band predictor")
                return None
                
        except Exception as e:
            logger.error(f"[{task_id}] Error in band filtering: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_multi_gpu(self, task: AudioBatch, num_gpus: int) -> List[Dict[str, Any]]:
        """
        Process items in parallel across multiple GPUs.
        
        Args:
            task: AudioBatch with multiple items
            num_gpus: Number of GPUs to use
            
        Returns:
            List of processed items that passed the band filter
        """
        total_items = len(task.data)
        model_path = self._resolve_model_path()
        
        # Prepare item data tuples for the worker function
        item_data_list = [
            (item, task.task_id, model_path, self.feature_group, 
             self.band_value, self.n_workers, self.feature_cache_size)
            for item in task.data
        ]
        
        # Get available GPU IDs
        available_gpus = list(range(num_gpus))
        
        results = []
        
        # Use ThreadPoolExecutor with GPU round-robin assignment
        max_workers = min(num_gpus, total_items)
        
        logger.info(f"[BandFilter] Using multi-GPU parallel processing: {max_workers} GPUs for {total_items} items")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, item_data in enumerate(item_data_list):
                gpu_id = available_gpus[i % num_gpus]  # Round-robin GPU assignment
                future = executor.submit(_band_process_on_gpu, item_data, gpu_id)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"[{task.task_id}] Error in multi-GPU Band processing: {e}")
        
        return results

    def process(self, task: AudioBatch) -> Optional[AudioBatch]:
        """
        Filter audio based on bandwidth classification.
        
        Automatically uses parallel processing when:
        - Multiple items in task.data (> 1)
        - Resources specify cpus > 1 (CPU parallelism via ThreadPoolExecutor)
        - Resources specify gpus > 1 (Multi-GPU parallelism for feature extraction)
        - Resources specify gpus > 0 and gpus <= 1 (Single GPU for feature extraction)
        
        The stage discovers its allocated resources via self._resources
        (set via .with_(resources=Resources(...))) and optimizes accordingly.
        
        Args:
            task: AudioBatch with waveform data
            
        Returns:
            AudioBatch if passes band filter, None otherwise
        """
        self._initialize_predictor()
        
        if self._predictor is None:
            logger.error("Band predictor not available")
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
            logger.debug(f"[BandFilter] Using CPU parallel processing with {max_workers} workers for {total_items} items")
            
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
                        logger.error(f"[{task.task_id}] Error in parallel Band processing: {e}")
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
        
        logger.info(f"[BandFilter] {task.task_id}: {passed_count}/{total_items} passed ({self.band_value}) [{mode}]")
        
        return AudioBatch(data=results, task_id=task.task_id, dataset_name=task.dataset_name)
