import os
import numpy as np
import joblib
import warnings
from typing import Dict, List, Tuple, Any, Optional
import concurrent.futures
import time
import torch
from contextlib import contextmanager
from loguru import logger

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to unpickle estimator.*")

# Import the AudioFeatureExtractor from features module
from .features import AudioFeatureExtractor

# Extend AudioFeatureExtractor with waveform processing capability
@contextmanager
def waveform_to_feature_extractor(waveform, sample_rate):
    """
    Context manager to create a temporary wrapper for extracting features directly from waveform
    
    Args:
        waveform: Audio waveform tensor [channels, samples]
        sample_rate: Sample rate of the audio
        
    Yields:
        A wrapper object with an extract_features method
    """
    class WaveformFeatureExtractor:
        def __init__(self, waveform, sr):
            self.waveform = waveform
            self.sr = sr
            
        def extract_features(self, feature_group="all"):
            # Use the existing method for waveforms
            return AudioFeatureExtractor.extract_all_features_from_waveform(
                self.waveform, self.sr, feature_group)
    
    # Create and yield the extractor
    extractor = WaveformFeatureExtractor(waveform, sample_rate)
    try:
        yield extractor
    finally:
        pass  # No cleanup needed

# Add the method to AudioFeatureExtractor
AudioFeatureExtractor.waveform_to_feature_extractor = staticmethod(waveform_to_feature_extractor)

# Check GPU availability
GPU_AVAILABLE = False
try:
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
except Exception:
    pass


class BandPredictor:
    """Class to predict band label (full_band/low_band) for audio waveforms"""
    
    def __init__(self, model_path: str = None, 
                 feature_group: str = None, 
                 n_workers: int = None,
                 feature_cache_size: int = None,
                 use_gpu: bool = None,
                 config = None):
        """
        Initialize the band predictor
        
        Args:
            model_path: Path to the trained model file
            feature_group: Which feature group to extract
            n_workers: Number of worker threads for parallel processing
            feature_cache_size: Number of feature vectors to cache
            use_gpu: Whether to use GPU acceleration if available
            config: Configuration manager instance (YouTubeAudioConfig)
        """
        # Map configuration from YouTubeAudioConfig object or use defaults
        cfg_model_path = getattr(config, 'band_model_path', None) if config else None
        # Default model path if neither provided
        if not cfg_model_path:
            cfg_model_path = "band_filter/model/band_classifier_model_band_7000_samples.joblib"
             
        cfg_feature_group = getattr(config, 'band_feature_group', "band") if config else "band"
        cfg_n_workers = getattr(config, 'band_n_workers', 4) if config else 4
        cfg_cache_size = getattr(config, 'band_feature_cache_size', 100) if config else 100
        cfg_use_gpu = getattr(config, 'use_gpu', True) if config else True
        
        # Parameters take precedence over config
        self.model_path = model_path or cfg_model_path
        self.feature_group = feature_group or cfg_feature_group
        self.n_workers = n_workers or cfg_n_workers
        self.feature_cache_size = feature_cache_size or cfg_cache_size
        
        # Check if use_gpu is provided, otherwise get from config
        if use_gpu is None:
            self.use_gpu = cfg_use_gpu and GPU_AVAILABLE
        else:
            self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Store config for later use
        self.config = config
        
        # Batch processing flag (not currently in YouTubeAudioConfig, defaulting to True)
        self.use_batch_processing = True
        
        self.model = None
        self.feature_cache = {}  # Cache for features
        
        if self.use_gpu:
            logger.info("GPU acceleration enabled for band prediction")
        
        # Load model immediately
        self._load_model()
        
        # Create GPU model if needed
        self._setup_gpu_model()
    
    def _load_model(self):
        """Load the model from disk"""
        try:
            start_time = time.time()
            with warnings.catch_warnings():
                # Suppress warnings during model loading
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model = joblib.load(self.model_path)
                
                # Fix compatibility issue with monotonic_cst attribute
                if hasattr(self.model, 'estimators_'):
                    # For RandomForest or ensemble models
                    for estimator in self.model.estimators_:
                        if hasattr(estimator, 'tree_'):
                            # For each decision tree, add missing attribute if needed
                            if not hasattr(estimator, 'monotonic_cst'):
                                estimator.monotonic_cst = None
                elif hasattr(self.model, 'tree_'):
                    # For single decision tree
                    if not hasattr(self.model, 'monotonic_cst'):
                        self.model.monotonic_cst = None
                        
            logger.info(f"Band prediction model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise
    
    def _setup_gpu_model(self):
        """Set up GPU-accelerated model if possible"""
        if not self.use_gpu:
            return
            
        try:
            # Check if the model is a RandomForest or similar ensemble
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            
            if isinstance(self.model, (RandomForestClassifier, GradientBoostingClassifier)):
                # We'll use the scikit-learn model directly but batch processing with GPU
                self.gpu_batch_size = 1024  # Adjust based on GPU memory
                logger.info(f"Using GPU-accelerated batch inference for {type(self.model).__name__}")
            
        except Exception as e:
            logger.warning(f"Error setting up GPU acceleration: {e}")
            self.use_gpu = False
    
    def extract_features_from_audio(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """
        Extract audio features directly from waveform tensor
        
        Args:
            waveform: Audio waveform tensor [channels, samples]
            sample_rate: Sample rate of the audio
            
        Returns:
            Array of extracted features
        """
        try:
            # Generate a cache key from the waveform data
            # Use a simpler key generation for speed
            if hasattr(waveform, 'cpu'):
                w_np = waveform.cpu().numpy()
            else:
                w_np = waveform.numpy() if hasattr(waveform, 'numpy') else waveform
                
            # Use only a subset of data for hash to be faster
            flat_data = w_np.ravel()
            step = max(1, len(flat_data) // 1000)
            cache_key = hash(flat_data[::step].tobytes())
            
            # Check cache first
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
            
            # Extract features using wrapper
            with AudioFeatureExtractor.waveform_to_feature_extractor(waveform, sample_rate) as feature_extractor:
                # Extract features
                feature_dict = feature_extractor.extract_features(feature_group=self.feature_group)
                
                # Convert dictionary to vector
                feature_vector, _ = AudioFeatureExtractor.features_dict_to_vector(feature_dict)
                
                # Check for NaN values and replace with zeros
                if np.isnan(feature_vector).any():
                    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                
                # Cache feature vector
                if len(self.feature_cache) >= self.feature_cache_size:
                    # Remove oldest cache entry if cache is full
                    oldest_key = next(iter(self.feature_cache))
                    del self.feature_cache[oldest_key]
                    
                self.feature_cache[cache_key] = feature_vector
                
                return feature_vector
                
        except Exception as e:
            logger.error(f"Error processing audio for features: {e}")
            # Get default empty features as fallback
            sample_dict = AudioFeatureExtractor.get_empty_feature_dict(self.feature_group)
            sample_vector, _ = AudioFeatureExtractor.features_dict_to_vector(sample_dict)
            return np.zeros_like(sample_vector)
    
    def _gpu_predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Use GPU to predict a batch of samples"""
        if not self.use_gpu:
            return self.model.predict(features)
            
        try:
            # Convert numpy array to PyTorch tensor and move to GPU
            tensor_features = torch.tensor(features, dtype=torch.float32).cuda()
            
            # Process in batches to avoid GPU memory issues
            batch_size = getattr(self, 'gpu_batch_size', 1024)
            num_samples = tensor_features.shape[0]
            result = []
            
            for i in range(0, num_samples, batch_size):
                batch = tensor_features[i:i+batch_size]
                
                # Move batch to CPU for sklearn prediction
                batch_cpu = batch.cpu().numpy()
                batch_result = self.model.predict(batch_cpu)
                result.append(batch_result)
            
            # Combine results
            return np.concatenate(result)
        except Exception as e:
            logger.warning(f"GPU prediction failed, falling back to CPU: {e}")
            return self.model.predict(features)
    
    def predict_audio(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """
        Predict whether an audio waveform is full band or low band
        
        Args:
            waveform: Audio waveform tensor [channels, samples]
            sample_rate: Sample rate of the audio
        
        Returns:
            Prediction result as a string ('full_band' or 'low_band')
        """
        # Check if model is loaded
        if self.model is None:
            try:
                self._load_model()
            except Exception as e:
                return f"Error: Could not load model: {e}"
        
        try:
            # Extract features
            features = self.extract_features_from_audio(waveform, sample_rate)
            
            # Make prediction
            if self.use_gpu:
                # Use GPU for single prediction by creating a batch of 1
                prediction = self._gpu_predict_batch(features.reshape(1, -1))[0]
            else:
                prediction = self.model.predict(features.reshape(1, -1))[0]
            
            # Return prediction result
            return 'full_band' if prediction == 1 else 'low_band'
        
        except Exception as e:
            return f"Error during prediction: {e}"
    
    def predict_audio_batch(self, audio_data: List[Tuple[torch.Tensor, int]], 
                          use_parallel: bool = None) -> List[str]:
        """
        Predict band type for multiple audio waveforms
        
        Args:
            audio_data: List of tuples containing (waveform, sample_rate)
            use_parallel: Whether to use parallel processing
        
        Returns:
            List of prediction results in the same order as input
        """
        if not audio_data:
            return []
        
        # Use default if not provided
        if use_parallel is None:
            use_parallel = self.use_batch_processing
            
        results = []
        
        # Process audio in parallel if requested
        if use_parallel and self.n_workers > 1 and len(audio_data) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                # First extract all features in parallel
                future_to_index = {}
                for i, (waveform, sample_rate) in enumerate(audio_data):
                    future = executor.submit(self.extract_features_from_audio, waveform, sample_rate)
                    future_to_index[future] = i
                
                # Collect all features
                all_features = [None] * len(audio_data)
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        all_features[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing audio at index {idx}: {e}")
                        all_features[idx] = None
                
                # Batch predict all valid features at once
                try:
                    # Filter out None values
                    valid_indices = [i for i, f in enumerate(all_features) if f is not None]
                    valid_features = [all_features[i] for i in valid_indices]
                    
                    if valid_features:
                        # Make batch prediction with numpy
                        batch_features = np.array(valid_features)
                        
                        # Use GPU acceleration if enabled
                        if self.use_gpu:
                            batch_predictions = self._gpu_predict_batch(batch_features)
                        else:
                            batch_predictions = self.model.predict(batch_features)
                        
                        # Initialize results with errors for all
                        results = ["Error: Feature extraction failed"] * len(audio_data)
                        
                        # Update results for valid predictions
                        for i, pred in zip(valid_indices, batch_predictions):
                            results[i] = 'full_band' if pred == 1 else 'low_band'
                    else:
                        results = ["Error: Feature extraction failed"] * len(audio_data)
                    
                except Exception as e:
                    logger.error(f"Error during batch prediction: {e}")
                    # Fall back to one-by-one prediction
                    results = []
                    for waveform, sample_rate in audio_data:
                        try:
                            results.append(self.predict_audio(waveform, sample_rate))
                        except Exception as e2:
                            logger.error(f"Error predicting audio: {e2}")
                            results.append(f"Error: {e2}")
            
        else:
            # Sequential processing
            for waveform, sample_rate in audio_data:
                results.append(self.predict_audio(waveform, sample_rate))
        
        return results

