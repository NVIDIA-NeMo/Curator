import numpy as np
import librosa
from typing import Dict, List, Tuple
import pyloudnorm as pyln
from loguru import logger


class AudioFeatureExtractor:
    """Advanced audio feature extractor for audio quality classification"""
    
    @staticmethod
    def extract_all_features(file_path: str, sr: int = 48000, feature_group: str = "all") -> Dict[str, float]:
        """
        Extract a comprehensive set of audio features from a file
        
        Args:
            file_path: Path to the audio file
            sr: Sample rate to load the audio with
            feature_group: Which feature group to extract ("frequency", "noise", "temporal", "perceptual", "band", "all")
            
        Returns:
            Dictionary of feature names and values
        """
        try:
            # Load the audio file with librosa
            y, sr = librosa.load(file_path, sr=sr, mono=True)
            
            # Extract the requested feature groups
            all_features = {}
            
            if feature_group == "frequency" or feature_group == "all":
                frequency_features = AudioFeatureExtractor.extract_frequency_features(y, sr)
                all_features.update(frequency_features)
                
            if feature_group == "noise" or feature_group == "all":
                noise_features = AudioFeatureExtractor.extract_noise_features(y, sr)
                all_features.update(noise_features)
                
            if feature_group == "temporal" or feature_group == "all":
                temporal_features = AudioFeatureExtractor.extract_temporal_features(y, sr)
                all_features.update(temporal_features)
                
            if feature_group == "perceptual" or feature_group == "all":
                perceptual_features = AudioFeatureExtractor.extract_perceptual_features(y, sr)
                all_features.update(perceptual_features)
                
            if feature_group == "band" or feature_group == "all":
                band_energy = AudioFeatureExtractor.calculate_band_energy(y, sr)
                all_features.update(band_energy)
            
            # Check for NaN values and replace them
            for key in all_features:
                if np.isnan(all_features[key]):
                    logger.warning(f"NaN value found for feature {key} in {file_path}, replacing with 0")
                    all_features[key] = 0.0
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return AudioFeatureExtractor.get_empty_feature_dict(feature_group)
    
    @staticmethod
    def get_empty_feature_dict(feature_group: str = "all") -> Dict[str, float]:
        """
        Create an empty feature dictionary with all required keys set to 0.0
        
        Args:
            feature_group: Which feature group to include ("frequency", "noise", "temporal", "perceptual", "band", "all")
            
        Returns:
            Dictionary with all feature keys initialized to 0.0
        """
        # Create empty dictionaries for each feature group
        empty_dict = {}
        
        # Frequency features
        frequency_keys = [
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_flatness_mean', 'spectral_flatness_std',
            'spectral_entropy_mean', 'spectral_entropy_std'
        ]
        
        # Noise features
        noise_keys = [
            'estimated_snr', 'clipping_ratio', 'harmonic_to_noise_ratio'
        ]
        
        # Temporal features
        temporal_keys = [
            'zero_crossing_rate_mean', 'zero_crossing_rate_std',
            'rms_energy_mean', 'rms_energy_std',
            'envelope_mean', 'envelope_std', 'envelope_skewness', 'envelope_kurtosis',
            'avg_attack_time'
        ]
        
        # Perceptual features
        perceptual_keys = []
        # MFCCs
        for i in range(13):
            perceptual_keys.append(f'mfcc_{i+1}_mean')
            perceptual_keys.append(f'mfcc_{i+1}_std')
        
        # Chroma features
        perceptual_keys.append('chroma_mean')
        perceptual_keys.append('chroma_std')
        for i in range(12):  # 12 chroma bins
            perceptual_keys.append(f'chroma_{i+1}_mean')
        
        # Spectral contrast
        perceptual_keys.append('contrast_mean')
        perceptual_keys.append('contrast_std')
        for i in range(7):  # 7 contrast bands is typical
            perceptual_keys.append(f'contrast_band_{i+1}_mean')
        
        # Band energy features
        band_energy_keys = []
        bands = [
            'low1', 'low2', 'low3',
            'mid1', 'mid2', 'mid3', 'mid4', 'mid5', 'mid6', 'mid7', 'mid8', 'mid9', 'mid10',
            'high', 'high1', 'high2', 'high3', 'high4', 'high5', 'high6', 'high7', 'high8', 'high9', 'high10'
        ]
        for band in bands:
            band_energy_keys.append(f'band_energy_{band}')
        
        # Initialize selected keys based on feature_group
        all_keys = []
        if feature_group == "all":
            all_keys = frequency_keys + noise_keys + temporal_keys + perceptual_keys + band_energy_keys
        elif feature_group == "frequency":
            all_keys = frequency_keys
        elif feature_group == "noise":
            all_keys = noise_keys
        elif feature_group == "temporal":
            all_keys = temporal_keys
        elif feature_group == "perceptual":
            all_keys = perceptual_keys
        elif feature_group == "band":
            all_keys = band_energy_keys
        
        # Initialize all selected keys to 0.0
        for key in all_keys:
            empty_dict[key] = 0.0
            
        return empty_dict
    
    @staticmethod
    def extract_frequency_features(y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract frequency domain features
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of frequency features
        """
        features = {}
        
        try:
            # Compute the spectrogram
            n_fft = 2048
            S = np.abs(librosa.stft(y, n_fft=n_fft))
            
            # Spectral Centroid
            centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(centroid))
            features['spectral_centroid_std'] = float(np.std(centroid))
            
            # Spectral Bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
            features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(bandwidth))
            
            # Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
            features['spectral_rolloff_mean'] = float(np.mean(rolloff))
            features['spectral_rolloff_std'] = float(np.std(rolloff))
            
            # Spectral Flatness
            flatness = librosa.feature.spectral_flatness(S=S)
            features['spectral_flatness_mean'] = float(np.mean(flatness))
            features['spectral_flatness_std'] = float(np.std(flatness))
            
            # Spectral Entropy
            # Normalize spectrogram to calculate entropy
            S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-10)  # Avoid division by zero
            entropy = -np.sum(S_norm * np.log2(S_norm + 1e-10), axis=0)
            features['spectral_entropy_mean'] = float(np.mean(entropy))
            features['spectral_entropy_std'] = float(np.std(entropy))
        except Exception as e:
            logger.error(f"Error extracting frequency features: {e}")
            # Set default values for all features in this group
            keys = ['spectral_centroid_mean', 'spectral_centroid_std', 
                   'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                   'spectral_rolloff_mean', 'spectral_rolloff_std',
                   'spectral_flatness_mean', 'spectral_flatness_std',
                   'spectral_entropy_mean', 'spectral_entropy_std']
            for key in keys:
                features[key] = 0.0
            
        return features
    
    @staticmethod
    def extract_noise_features(y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract noise and distortion features
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of noise-related features
        """
        features = {}
        
        try:
            # Estimate Signal-to-Noise Ratio (SNR)
            # Using a simple heuristic: assume lowest 5% of frames are noise
            frame_length = 2048
            hop_length = 512
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
            
            # Calculate energy per frame
            frame_energy = np.sum(frames**2, axis=0)
            
            # Sort frames by energy
            sorted_idx = np.argsort(frame_energy)
            
            # Consider lowest 5% of frames (by energy) as noise
            noise_frames = sorted_idx[:int(0.05 * len(sorted_idx))]
            noise_energy = np.mean(frame_energy[noise_frames])
            
            # Calculate signal energy by excluding noise frames
            signal_energy = np.mean(frame_energy) - noise_energy
            
            # Compute SNR
            if noise_energy > 0:
                snr = 10 * np.log10(signal_energy / noise_energy)
                features['estimated_snr'] = float(snr)
            else:
                features['estimated_snr'] = 100.0  # High value if no noise detected
            
            # Detect clipping/saturation
            # Count samples that are at or very close to the max amplitude
            threshold = 0.99
            num_clipped = np.sum(np.abs(y) > threshold)
            features['clipping_ratio'] = float(num_clipped / len(y))
            
            # Harmonic-to-Noise estimation based on harmonic percussive separation
            # This is a simplification - true HNR would require pitch estimation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_energy = np.sum(y_harmonic**2)
            noise_energy = np.sum(y_percussive**2)
            
            if noise_energy > 0:
                hnr = 10 * np.log10(harmonic_energy / noise_energy)
                features['harmonic_to_noise_ratio'] = float(hnr)
            else:
                features['harmonic_to_noise_ratio'] = 100.0  # High value if no noise detected
        except Exception as e:
            logger.error(f"Error extracting noise features: {e}")
            features['estimated_snr'] = 50.0
            features['clipping_ratio'] = 0.0
            features['harmonic_to_noise_ratio'] = 50.0
            
        return features
    
    @staticmethod
    def extract_temporal_features(y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract time-domain features
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        try:
            # Zero-Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zero_crossing_rate_mean'] = float(np.mean(zcr))
            features['zero_crossing_rate_std'] = float(np.std(zcr))
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)
            features['rms_energy_mean'] = float(np.mean(rms))
            features['rms_energy_std'] = float(np.std(rms))
            
            # Temporal Envelope Statistics
            # Extract the envelope
            envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sr))
            
            # Envelope statistics
            envelope_mean = float(np.mean(envelope))
            envelope_std = float(np.std(envelope) + 1e-10)  # Avoid zero std
            
            features['envelope_mean'] = envelope_mean
            features['envelope_std'] = envelope_std
            features['envelope_skewness'] = float(np.mean((envelope - envelope_mean)**3) / (envelope_std**3))
            features['envelope_kurtosis'] = float(np.mean((envelope - envelope_mean)**4) / (envelope_std**4))
            
            # Calculate attack time - simplification
            # Attack time can be defined as how quickly the envelope reaches a peak from silence
            attack_times = []
            threshold = 0.2 * np.max(envelope) if len(envelope) > 0 and np.max(envelope) > 0 else 0
            
            i = 0
            while i < len(envelope):
                if envelope[i] > threshold:
                    start_idx = i
                    while i < len(envelope) and envelope[i] > threshold:
                        i += 1
                    end_idx = i
                    
                    # Find the peak in this region
                    if start_idx < end_idx:
                        peak_idx = start_idx + np.argmax(envelope[start_idx:end_idx])
                        
                        # Calculate the attack time (in frames)
                        attack_time = peak_idx - start_idx
                        attack_times.append(attack_time)
                i += 1
            
            features['avg_attack_time'] = float(np.mean(attack_times) if attack_times else 0)
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            # Set default values
            keys = ['zero_crossing_rate_mean', 'zero_crossing_rate_std',
                   'rms_energy_mean', 'rms_energy_std',
                   'envelope_mean', 'envelope_std', 'envelope_skewness', 'envelope_kurtosis',
                   'avg_attack_time']
            for key in keys:
                features[key] = 0.0
            
        return features
    
    @staticmethod
    def extract_perceptual_features(y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract perceptual audio features
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of perceptual features
        """
        features = {}
        
        try:
            # MFCCs (Mel-Frequency Cepstral Coefficients)
            n_mfcc = 13  # Typical value
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Store each MFCC coefficient's mean and std
            for i in range(n_mfcc):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
            # Chroma Features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            # For each chroma bin
            for i in range(chroma.shape[0]):
                features[f'chroma_{i+1}_mean'] = float(np.mean(chroma[i]))
            
            # Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # Overall statistics
            features['contrast_mean'] = float(np.mean(contrast))
            features['contrast_std'] = float(np.std(contrast))
            
            # For each contrast band
            for i in range(contrast.shape[0]):
                features[f'contrast_band_{i+1}_mean'] = float(np.mean(contrast[i]))
        except Exception as e:
            logger.error(f"Error extracting perceptual features: {e}")
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = 0.0
                features[f'mfcc_{i+1}_std'] = 0.0
            
            # Set default values for chroma features
            features['chroma_mean'] = 0.0
            features['chroma_std'] = 0.0
            for i in range(12):  # 12 chroma bins
                features[f'chroma_{i+1}_mean'] = 0.0
            
            # Set default values for contrast features
            features['contrast_mean'] = 0.0
            features['contrast_std'] = 0.0
            for i in range(7):  # 7 contrast bands
                features[f'contrast_band_{i+1}_mean'] = 0.0
            
        return features
    
    @staticmethod
    def calculate_band_energy(y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Calculate energy in different frequency bands with LUFS normalization
        
        Args:
            y: Audio time series
            sr: Sampling rate
            
        Returns:
            Dictionary with energy levels for each frequency band
        """
        # Define bands
        bands = {
            'low1': (0, 1000),
            'low2': (1000, 2000),
            'low3': (2000, 3000),
            'mid1': (3000, 4000),
            'mid2': (4000, 5000),
            'mid3': (5000, 6000),
            'mid4': (6000, 7000),
            'mid5': (7000, 8000),
            'mid6': (8000, 9000),
            'mid7': (9000, 10000),
            'mid8': (10000, 11000),
            'mid9': (11000, 12000),
            'mid10': (12000, 13000),
            'high': (13000, 14000),
            'high1': (14000, 15000),
            'high2': (15000, 16000),
            'high3': (16000, 17000),
            'high4': (17000, 18000),
            'high5': (18000, 19000),
            'high6': (19000, 20000),
            'high7': (20000, 21000),
            'high8': (21000, 22000),
            'high9': (22000, 23000),
            'high10': (23000, 24000)
        }
        
        band_energy = {}
        
        try:
            # Make sure audio is mono
            if y.ndim > 1 and y.shape[0] > 1:
                # Average across channels to get mono
                y = np.mean(y, axis=0)
            
            # Ensure audio is one-dimensional for PyLoudNorm
            if y.ndim > 1:
                y = y.squeeze()
                
            # Normalize audio to -25 LUFS before processing
            # Create BS.1770 meter
            meter = pyln.Meter(sr)
            
            # Measure original loudness
            original_loudness = meter.integrated_loudness(y)
            
            # Normalize to -25 LUFS if audio is not silent
            if original_loudness > -100.0:  # Check if audio is not essentially silent
                normalized_audio = pyln.normalize.loudness(y, original_loudness, -25.0)
            else:
                normalized_audio = y  # Keep as is if it's silent
            
            # STFT and power spectrum of normalized audio
            n_fft = 4096
            D = np.abs(librosa.stft(normalized_audio, n_fft=n_fft))
            power = D ** 2
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Find global max power for reference (across all bands)
            global_max_power = np.max(power) if np.max(power) > 0 else 1e-10
            
            # Calculate mean energy in dB for each band
            for band, (f_min, f_max) in bands.items():
                mask = (freqs >= f_min) & (freqs < f_max)
                if np.any(mask):  # Check if any frequencies fall in this band
                    mean_power = np.mean(power[mask, :])
                    # Use global max power as reference for all bands for consistent scaling
                    band_energy[f'band_energy_{band}'] = float(librosa.power_to_db(mean_power, ref=global_max_power))
                    
                    # Apply additional attenuation to higher frequency bands to better represent
                    # natural roll-off in human hearing and typical audio content
                    if f_min >= 10000:
                        # Progressively increase attenuation with frequency
                        attenuation_factor = (f_min - 10000) / 14000 * 12  # Up to 12dB more attenuation at highest bands
                        band_energy[f'band_energy_{band}'] -= attenuation_factor
                else:
                    # Lower default for empty bands
                    band_energy[f'band_energy_{band}'] = -120.0  # Much lower default to better represent absence of energy
        except Exception as e:
            logger.error(f"Error calculating band energy: {e}")
            # Set default values for all band energies
            for band in bands:
                band_energy[f'band_energy_{band}'] = -120.0
            
        return band_energy
    
    @staticmethod
    def features_dict_to_vector(features_dict: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert a dictionary of features to a feature vector
        
        Args:
            features_dict: Dictionary of feature name-value pairs
            
        Returns:
            Tuple of (feature_vector, feature_names)
        """
        # Check for empty dictionary
        if not features_dict:
            # Return a default empty dictionary
            features_dict = AudioFeatureExtractor.get_empty_feature_dict()
            
        # Sort features by name for consistency
        feature_names = sorted(features_dict.keys())
        
        # Create vector, ensuring no NaN values
        feature_vector = []
        for name in feature_names:
            value = features_dict[name]
            # Convert any NaN to 0.0
            if np.isnan(value):
                logger.warning(f"NaN value found for feature {name}, replacing with 0")
                value = 0.0
            feature_vector.append(value)
            
        return np.array(feature_vector), feature_names
    
    @staticmethod
    def extract_all_features_from_waveform(waveform, sr: int, feature_group: str = "all") -> Dict[str, float]:
        """
        Extract a comprehensive set of audio features from a waveform tensor/array
        
        Args:
            waveform: Audio waveform tensor/array
            sr: Sample rate of the waveform
            feature_group: Which feature group to extract ("frequency", "noise", "temporal", "perceptual", "band", "all")
            
        Returns:
            Dictionary of feature names and values
        """
        try:
            # Convert to numpy if it's a tensor
            if hasattr(waveform, 'numpy'):
                y = waveform.numpy()
                # Handle multi-channel audio
                if y.ndim > 1 and y.shape[0] > 1:
                    y = np.mean(y, axis=0)
            else:
                y = waveform
                # Handle multi-channel audio
                if y.ndim > 1 and y.shape[0] > 1:
                    y = np.mean(y, axis=0)
            
            # Extract the requested feature groups
            all_features = {}
            
            if feature_group == "frequency" or feature_group == "all":
                frequency_features = AudioFeatureExtractor.extract_frequency_features(y, sr)
                all_features.update(frequency_features)
                
            if feature_group == "noise" or feature_group == "all":
                noise_features = AudioFeatureExtractor.extract_noise_features(y, sr)
                all_features.update(noise_features)
                
            if feature_group == "temporal" or feature_group == "all":
                temporal_features = AudioFeatureExtractor.extract_temporal_features(y, sr)
                all_features.update(temporal_features)
                
            if feature_group == "perceptual" or feature_group == "all":
                perceptual_features = AudioFeatureExtractor.extract_perceptual_features(y, sr)
                all_features.update(perceptual_features)
                
            if feature_group == "band" or feature_group == "all":
                band_energy = AudioFeatureExtractor.calculate_band_energy(y, sr)
                all_features.update(band_energy)
            
            # Check for NaN values and replace them
            for key in all_features:
                if np.isnan(all_features[key]):
                    logger.warning(f"NaN value found for feature {key}, replacing with 0")
                    all_features[key] = 0.0
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error processing waveform: {e}")
            # Return empty feature dict in case of error
            return AudioFeatureExtractor.get_empty_feature_dict(feature_group) 