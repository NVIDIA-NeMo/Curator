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

    def process(self, task: AudioBatch) -> Optional[AudioBatch]:
        """
        Filter audio based on bandwidth classification.
        
        Automatically uses parallel processing when:
        - Multiple items in task.data (> 1)
        - Resources specify cpus > 1 (CPU parallelism via ThreadPoolExecutor)
        - Resources specify gpus > 0 (GPU is used for feature extraction)
        
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
            return None
        
        total_items = len(task.data)
        
        # Determine processing strategy based on resources and item count
        use_cpu_parallel = (
            total_items > 1 and 
            self._resources.cpus > 1 and 
            self._resources.gpus == 0  # CPU parallelism only when not using GPU
        )
        
        if use_cpu_parallel:
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
        else:
            # Sequential processing (default, or when GPU is used)
            results = []
            for item in task.data:
                result = self._process_single_item(item, task.task_id)
                if result is not None:
                    results.append(result)
        
        passed_count = len(results)
        rejected_count = total_items - passed_count
        
        mode = "CPU parallel" if use_cpu_parallel else ("GPU" if self._resources.gpus > 0 else "sequential")
        logger.info(f"[BandFilter] {task.task_id}: {passed_count}/{total_items} passed ({self.band_value}) [{mode}]")
        
        if not results:
            return None
        
        return AudioBatch(data=results, task_id=task.task_id, dataset_name=task.dataset_name)
