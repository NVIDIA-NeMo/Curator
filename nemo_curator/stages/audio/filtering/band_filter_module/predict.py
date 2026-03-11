import numpy as np
import joblib
import warnings
from typing import Dict, List, Tuple, Any, Optional
import concurrent.futures
import time
import torch
from loguru import logger

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to unpickle estimator.*")

from .features import AudioFeatureExtractor

# Check GPU availability
GPU_AVAILABLE = False
try:
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
except Exception:
    pass


class BandPredictor:
    """Class to predict band label (full_band/narrow_band) for audio waveforms."""

    def __init__(self, model_path: str = None,
                 n_workers: int = None,
                 feature_cache_size: int = None,
                 use_gpu: bool = None,
                 config=None):
        """
        Initialize the band predictor.

        Args:
            model_path: Path to the trained model file
            n_workers: Number of worker threads for parallel processing
            feature_cache_size: Number of feature vectors to cache
            use_gpu: Whether to use GPU acceleration if available
            config: Configuration manager instance
        """
        cfg_model_path = getattr(config, 'band_model_path', None) if config else None
        if not cfg_model_path:
            cfg_model_path = "band_filter/model/band_classifier_model_band_7000_samples.joblib"

        cfg_n_workers = getattr(config, 'band_n_workers', 4) if config else 4
        cfg_cache_size = getattr(config, 'band_feature_cache_size', 100) if config else 100
        cfg_use_gpu = getattr(config, 'use_gpu', True) if config else True

        self.model_path = model_path or cfg_model_path
        self.n_workers = n_workers or cfg_n_workers
        self.feature_cache_size = feature_cache_size or cfg_cache_size

        if use_gpu is None:
            self.use_gpu = cfg_use_gpu and GPU_AVAILABLE
        else:
            self.use_gpu = use_gpu and GPU_AVAILABLE

        self.config = config
        self.use_batch_processing = True
        self.model = None
        self.feature_cache = {}

        if self.use_gpu:
            logger.info("GPU acceleration enabled for band prediction")

        self._load_model()
        self._setup_gpu_model()

    def _load_model(self):
        """Load the model from disk."""
        try:
            start_time = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model = joblib.load(self.model_path)

                if hasattr(self.model, 'estimators_'):
                    for estimator in self.model.estimators_:
                        if hasattr(estimator, 'tree_'):
                            if not hasattr(estimator, 'monotonic_cst'):
                                estimator.monotonic_cst = None
                elif hasattr(self.model, 'tree_'):
                    if not hasattr(self.model, 'monotonic_cst'):
                        self.model.monotonic_cst = None

            logger.info(f"Band prediction model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise

    def _setup_gpu_model(self):
        """Set up GPU-accelerated model if possible."""
        if not self.use_gpu:
            return

        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

            if isinstance(self.model, (RandomForestClassifier, GradientBoostingClassifier)):
                self.gpu_batch_size = 1024
                logger.info(f"Using GPU-accelerated batch inference for {type(self.model).__name__}")

        except Exception as e:
            logger.warning(f"Error setting up GPU acceleration: {e}")
            self.use_gpu = False

    def extract_features_from_audio(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """
        Extract band energy features directly from waveform tensor.

        Args:
            waveform: Audio waveform tensor [channels, samples]
            sample_rate: Sample rate of the audio

        Returns:
            Array of extracted features
        """
        try:
            if hasattr(waveform, 'cpu'):
                w_np = waveform.cpu().numpy()
            else:
                w_np = waveform.numpy() if hasattr(waveform, 'numpy') else waveform

            flat_data = w_np.ravel()
            step = max(1, len(flat_data) // 1000)
            cache_key = hash(flat_data[::step].tobytes())

            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]

            feature_dict = AudioFeatureExtractor.extract_band_features_from_waveform(waveform, sample_rate)
            feature_vector, _ = AudioFeatureExtractor.features_dict_to_vector(feature_dict)

            if np.isnan(feature_vector).any():
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)

            if len(self.feature_cache) >= self.feature_cache_size:
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]

            self.feature_cache[cache_key] = feature_vector

            return feature_vector

        except Exception as e:
            logger.error(f"Error processing audio for features: {e}")
            sample_dict = AudioFeatureExtractor.get_empty_feature_dict()
            sample_vector, _ = AudioFeatureExtractor.features_dict_to_vector(sample_dict)
            return np.zeros_like(sample_vector)

    def _gpu_predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Use GPU to predict a batch of samples."""
        if not self.use_gpu:
            return self.model.predict(features)

        try:
            tensor_features = torch.tensor(features, dtype=torch.float32).cuda()

            batch_size = getattr(self, 'gpu_batch_size', 1024)
            num_samples = tensor_features.shape[0]
            result = []

            for i in range(0, num_samples, batch_size):
                batch = tensor_features[i:i+batch_size]
                batch_cpu = batch.cpu().numpy()
                batch_result = self.model.predict(batch_cpu)
                result.append(batch_result)

            return np.concatenate(result)
        except Exception as e:
            logger.warning(f"GPU prediction failed, falling back to CPU: {e}")
            return self.model.predict(features)

    def predict_audio(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """
        Predict whether an audio waveform is full band or narrow band.

        Args:
            waveform: Audio waveform tensor [channels, samples]
            sample_rate: Sample rate of the audio

        Returns:
            Prediction result as a string ('full_band' or 'narrow_band')
        """
        if self.model is None:
            try:
                self._load_model()
            except Exception as e:
                return f"Error: Could not load model: {e}"

        try:
            features = self.extract_features_from_audio(waveform, sample_rate)

            if self.use_gpu:
                prediction = self._gpu_predict_batch(features.reshape(1, -1))[0]
            else:
                prediction = self.model.predict(features.reshape(1, -1))[0]

            return 'full_band' if prediction == 1 else 'narrow_band'

        except Exception as e:
            return f"Error during prediction: {e}"

    def predict_audio_batch(self, audio_data: List[Tuple[torch.Tensor, int]],
                            use_parallel: bool = None) -> List[str]:
        """
        Predict band type for multiple audio waveforms.

        Args:
            audio_data: List of tuples containing (waveform, sample_rate)
            use_parallel: Whether to use parallel processing

        Returns:
            List of prediction results in the same order as input
        """
        if not audio_data:
            return []

        if use_parallel is None:
            use_parallel = self.use_batch_processing

        results = []

        if use_parallel and self.n_workers > 1 and len(audio_data) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_index = {}
                for i, (waveform, sample_rate) in enumerate(audio_data):
                    future = executor.submit(self.extract_features_from_audio, waveform, sample_rate)
                    future_to_index[future] = i

                all_features = [None] * len(audio_data)
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        all_features[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing audio at index {idx}: {e}")
                        all_features[idx] = None

                try:
                    valid_indices = [i for i, f in enumerate(all_features) if f is not None]
                    valid_features = [all_features[i] for i in valid_indices]

                    if valid_features:
                        batch_features = np.array(valid_features)

                        if self.use_gpu:
                            batch_predictions = self._gpu_predict_batch(batch_features)
                        else:
                            batch_predictions = self.model.predict(batch_features)

                        results = ["Error: Feature extraction failed"] * len(audio_data)

                        for i, pred in zip(valid_indices, batch_predictions):
                            results[i] = 'full_band' if pred == 1 else 'narrow_band'
                    else:
                        results = ["Error: Feature extraction failed"] * len(audio_data)

                except Exception as e:
                    logger.error(f"Error during batch prediction: {e}")
                    results = []
                    for waveform, sample_rate in audio_data:
                        try:
                            results.append(self.predict_audio(waveform, sample_rate))
                        except Exception as e2:
                            logger.error(f"Error predicting audio: {e2}")
                            results.append(f"Error: {e2}")

        else:
            for waveform, sample_rate in audio_data:
                results.append(self.predict_audio(waveform, sample_rate))

        return results
