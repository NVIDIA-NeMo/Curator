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
SIGMOS (Signal-based Mean Opinion Score) filter stage.

Filters audio segments based on SIGMOS quality metrics including
noise, overall quality, signal quality, coloration, discontinuity,
loudness, and reverberation.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
    from nemo_curator.stages.resources import Resources
    
    pipeline = Pipeline(name="quality_pipeline")
    pipeline.add_stage(
        SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        .with_(resources=Resources(gpus=0.3))
    )
"""

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from pydub import AudioSegment

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from ..configs import SIGMOSConfig


def _load_audio_as_pydub(audio_path: str) -> AudioSegment:
    """
    Load audio file as PyDub AudioSegment.
    
    Supports standalone usage of stages without requiring previous stages.
    """
    return AudioSegment.from_wav(audio_path)


@dataclass
class SIGMOSFilterStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    SIGMOS quality assessment filter stage.
    
    Filters audio segments based on SIGMOS quality metrics.
    Returns None for segments that don't meet threshold requirements.
    
    SIGMOS predicts (1-5 scale):
    - NOISE: Background noise level (higher = less noisy)
    - OVRL: Overall quality
    - SIG: Signal quality
    - COL: Coloration
    - DISC: Discontinuity
    - LOUD: Loudness
    - REVERB: Reverberation (higher = less reverb)
    
    Args:
        config: SIGMOSConfig object (overrides other params if provided)
        model_path: Path to SIGMOS ONNX model
        noise_threshold: Minimum noise score (None to disable)
        ovrl_threshold: Minimum overall score (None to disable)
        sig_threshold: Minimum signal score (None to disable)
        col_threshold: Minimum coloration score (None to disable)
        disc_threshold: Minimum discontinuity score (None to disable)
        loud_threshold: Minimum loudness score (None to disable)
        reverb_threshold: Minimum reverb score (None to disable)
    
    Note:
        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    
    Example:
        # Using config
        config = SIGMOSConfig(noise_threshold=4.0)
        stage = SIGMOSFilterStage(config=config)
        
        # Using parameters
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
    """
    
    config: Optional[SIGMOSConfig] = None
    model_path: str = "model/model-sigmos_1697718653_41d092e8-epo-200.onnx"
    noise_threshold: Optional[float] = 4.0
    ovrl_threshold: Optional[float] = 3.5
    sig_threshold: Optional[float] = None
    col_threshold: Optional[float] = None
    disc_threshold: Optional[float] = None
    loud_threshold: Optional[float] = None
    reverb_threshold: Optional[float] = None
    
    name: str = "SIGMOSFilter"
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
            self.noise_threshold = self.config.noise_threshold
            self.ovrl_threshold = self.config.ovrl_threshold
            self.sig_threshold = self.config.sig_threshold
            self.col_threshold = self.config.col_threshold
            self.disc_threshold = self.config.disc_threshold
            self.loud_threshold = self.config.loud_threshold
            self.reverb_threshold = self.config.reverb_threshold
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
        return [], ["sigmos_noise", "sigmos_ovrl", "sigmos_sig", "sigmos_col", 
                    "sigmos_disc", "sigmos_loud", "sigmos_reverb"]
    
    def setup(self, worker_metadata=None) -> None:
        """Load SIGMOS model on worker initialization."""
        self._initialize_model()
    
    def teardown(self) -> None:
        """Clean up resources."""
        if self._predict_function is not None:
            self._predict_function = None
            import torch
            torch.cuda.empty_cache()
    
    def _initialize_model(self):
        """Initialize the SIGMOS prediction function (thread-safe)."""
        if self._predict_function is None:
            with self._get_lock():
                if self._predict_function is None:
                    try:
                        # Import from local sigmos_filter_module (same directory)
                        from nemo_curator.stages.audio.filtering.sigmos_filter_module.sigmos_pipeline import predict_batch_mos
                        self._predict_function = predict_batch_mos
                        logger.info("SIGMOS prediction function loaded successfully")
                    except ImportError as e:
                        logger.error(f"Failed to import SIGMOS module: {e}")
                        raise
    
    def _resolve_model_path(self) -> str:
        """Resolve model path to absolute path."""
        if os.path.isabs(self.model_path):
            return self.model_path
        
        # Try relative to sigmos_filter_module first (default location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        module_dir = os.path.join(current_dir, 'sigmos_filter_module')
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
        """Process a single audio item and return it with SIGMOS scores if it passes thresholds.
        
        Returns the item with scores on success, None if it fails thresholds
        or on infrastructure errors (rejected).
        """
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
            pipeline_config = {'model_path': model_path}
            
            # Use current CUDA device (set by executor)
            if torch.cuda.is_available():
                selected_gpu = torch.cuda.current_device()
            else:
                selected_gpu = 0
            
            scores = self._predict_function(
                [temp_path], gpu_id=selected_gpu, config=pipeline_config
            )
            
            if temp_path not in scores:
                logger.warning(f"[{task_id}] No SIGMOS score returned")
                return None
            
            score_data = scores[temp_path]
            
            if isinstance(score_data, dict):
                noise = score_data.get('MOS_NOISE', 0)
                ovrl = score_data.get('MOS_OVRL', 0)
                sig = score_data.get('MOS_SIG', 0)
                col = score_data.get('MOS_COL', 0)
                disc = score_data.get('MOS_DISC', 0)
                loud = score_data.get('MOS_LOUD', 0)
                reverb = score_data.get('MOS_REVERB', 0)
            else:
                ovrl = float(score_data)
                noise = sig = col = disc = loud = reverb = 0
            
            # Log the actual scores for debugging
            logger.debug(
                f"[{task_id}] SIGMOS scores: NOISE={noise:.3f}, OVRL={ovrl:.3f}, SIG={sig:.3f}, "
                f"COL={col:.3f}, DISC={disc:.3f}, LOUD={loud:.3f}, REVERB={reverb:.3f}"
            )
            
            # Check thresholds
            passed = True
            fail_reasons = []
            if self.noise_threshold is not None and noise < self.noise_threshold:
                passed = False
                fail_reasons.append(f"NOISE {noise:.3f} < {self.noise_threshold}")
            if self.ovrl_threshold is not None and ovrl < self.ovrl_threshold:
                passed = False
                fail_reasons.append(f"OVRL {ovrl:.3f} < {self.ovrl_threshold}")
            if self.sig_threshold is not None and sig < self.sig_threshold:
                passed = False
                fail_reasons.append(f"SIG {sig:.3f} < {self.sig_threshold}")
            if self.col_threshold is not None and col < self.col_threshold:
                passed = False
                fail_reasons.append(f"COL {col:.3f} < {self.col_threshold}")
            if self.disc_threshold is not None and disc < self.disc_threshold:
                passed = False
                fail_reasons.append(f"DISC {disc:.3f} < {self.disc_threshold}")
            if self.loud_threshold is not None and loud < self.loud_threshold:
                passed = False
                fail_reasons.append(f"LOUD {loud:.3f} < {self.loud_threshold}")
            if self.reverb_threshold is not None and reverb < self.reverb_threshold:
                passed = False
                fail_reasons.append(f"REVERB {reverb:.3f} < {self.reverb_threshold}")
            
            if not passed:
                logger.info(f"[{task_id}] SIGMOS FAILED: {', '.join(fail_reasons)}")
            
            if passed:
                item['sigmos_noise'] = noise
                item['sigmos_ovrl'] = ovrl
                item['sigmos_sig'] = sig
                item['sigmos_col'] = col
                item['sigmos_disc'] = disc
                item['sigmos_loud'] = loud
                item['sigmos_reverb'] = reverb
                return item
            else:
                return None
                
        except Exception as e:
            logger.error(f"[{task_id}] Error in SIGMOS filtering: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def process(self, task: AudioBatch) -> Optional[AudioBatch]:
        """
        Filter audio based on SIGMOS quality scores.
        
        Automatically uses parallel processing when:
        - Multiple items in task.data (> 1)
        - Resources specify cpus > 1 (CPU parallelism via ThreadPoolExecutor)
        - Resources specify gpus > 0 (GPU batching for model inference)
        
        The stage discovers its allocated resources via self._resources
        (set via .with_(resources=Resources(...))) and optimizes accordingly.
        
        Args:
            task: AudioBatch with audio data
            
        Returns:
            AudioBatch with SIGMOS scores if passes, None otherwise
        """
        self._initialize_model()
        
        if self._predict_function is None:
            logger.error("SIGMOS prediction function not available")
            return AudioBatch(data=[], task_id=task.task_id, dataset_name=task.dataset_name)
        
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
            logger.debug(f"[SIGMOSFilter] Using CPU parallel processing with {max_workers} workers for {total_items} items")
            
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
                        logger.error(f"[{task.task_id}] Error in parallel SIGMOS processing: {e}")
        else:
            # Sequential processing (default, or when GPU is used - GPU handles batching internally)
            results = []
            for item in task.data:
                result = self._process_single_item(item, task.task_id)
                if result is not None:
                    results.append(result)
        
        passed_count = len(results)
        rejected_count = total_items - passed_count
        
        # Log summary
        threshold_parts = []
        if self.noise_threshold is not None:
            threshold_parts.append(f"NOISE>={self.noise_threshold}")
        if self.ovrl_threshold is not None:
            threshold_parts.append(f"OVRL>={self.ovrl_threshold}")
        if self.sig_threshold is not None:
            threshold_parts.append(f"SIG>={self.sig_threshold}")
        if self.col_threshold is not None:
            threshold_parts.append(f"COL>={self.col_threshold}")
        if self.disc_threshold is not None:
            threshold_parts.append(f"DISC>={self.disc_threshold}")
        if self.loud_threshold is not None:
            threshold_parts.append(f"LOUD>={self.loud_threshold}")
        if self.reverb_threshold is not None:
            threshold_parts.append(f"REVERB>={self.reverb_threshold}")
        threshold_str = ", ".join(threshold_parts) if threshold_parts else "none"
        
        mode = "CPU parallel" if use_cpu_parallel else ("GPU" if self._resources.gpus > 0 else "sequential")
        logger.info(f"[SIGMOSFilter] {task.task_id}: {passed_count}/{total_items} passed (thresholds: {threshold_str}) [{mode}]")
        
        return AudioBatch(data=results, task_id=task.task_id, dataset_name=task.dataset_name)
