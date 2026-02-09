#!/usr/bin/env python3
"""
SIGMOS Pipeline: A module for processing WAV files to get MOS predictions in pipelines.
This standalone module can be imported and used in other scripts.
"""

import os
import sys
import torch
import pandas as pd
import time
import copy
from pathlib import Path
import librosa
import numpy as np
from loguru import logger

# Import from third_party SIGMOS
try:
    from .third_party.sigmos.sigmos import build_sigmos_model
except ImportError:
    # Fallback for when running directly or tests
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, 'third_party'))
    from sigmos.sigmos import build_sigmos_model

# Model cache for storing initialized models by GPU ID
_MODEL_CACHE = {}


def get_config_val(config, key, default=None):
    """Helper to get config value from either dictionary or object."""
    if config is None:
        return default
        
    # 1. Check for nested dictionary: config['sigmos'][key]
    if isinstance(config, dict) and 'sigmos' in config and isinstance(config['sigmos'], dict):
        val = config['sigmos'].get(key)
        if val is not None:
            return val
            
    # 2. Check for flat dictionary key: config['sigmos_key']
    if isinstance(config, dict):
        val = config.get(f"sigmos_{key}")
        if val is not None:
            return val
            
    # 3. Check for object attribute: config.sigmos_key
    if hasattr(config, f"sigmos_{key}"):
        return getattr(config, f"sigmos_{key}")
        
    # 4. Check for object attribute: config.key (fallback)
    if hasattr(config, key):
        return getattr(config, key)

    return default


def build_sigmos_model_wrapper(config=None):
    """
    Build and return a SIGMOS model instance
    
    Args:
        config: Optional configuration object
    
    Returns:
        SIGMOS model instance
    """
    # Use config if provided, otherwise use defaults
    # Determine GPU usage
    use_gpu = True
    if config:
        if isinstance(config, dict):
            use_gpu = config.get('use_gpu', True) # Global toggle
        elif hasattr(config, 'use_gpu'):
            use_gpu = config.use_gpu
            
    force_cpu = not use_gpu
    device_id = get_config_val(config, 'gpu_id', 0)
    model_path = get_config_val(config, 'model_path', None)
    
    return build_sigmos_model(force_cpu=force_cpu, device_id=device_id, model_path=model_path)


def get_wav_files_from_path(input_path):
    """
    Get list of WAV files from a path (file or directory).
    
    Args:
        input_path (str): Path to file or directory
        
    Returns:
        list: List of WAV file paths
    """
    wav_files = []
    input_path = os.path.abspath(os.path.expanduser(input_path))
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith('.wav'):
            wav_files.append(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
    
    return wav_files


class SIGMOSPipeline:
    """
    A class for predicting MOS scores for audio files that can be used in pipelines.
    """
    def __init__(self, gpu_id=0, config=None):
        """
        Initialize the MOS predictor.
        
        Args:
            gpu_id (int): GPU ID to use
            config: Optional configuration object
        """
        self.gpu_id = gpu_id
        self.config = config
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
        
        # Get force_cpu setting from config if available
        use_gpu = True
        if config:
            if isinstance(config, dict):
                use_gpu = config.get('use_gpu', True)
            elif hasattr(config, 'use_gpu'):
                use_gpu = config.use_gpu
        self.force_cpu = not use_gpu
        
        # Get model path
        model_path = get_config_val(config, 'model_path', None)
        
        # Initialize model or get from cache
        global _MODEL_CACHE
        # Include model_path in cache key if provided to handle different models
        model_key_part = f"_{model_path}" if model_path else ""
        cache_key = f"cpu{model_key_part}" if self.force_cpu else f"gpu_{self.gpu_id}{model_key_part}"
        
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = build_sigmos_model(
                force_cpu=self.force_cpu, 
                device_id=self.gpu_id,
                model_path=model_path
            )
        
        self.model = _MODEL_CACHE[cache_key]
    
    def predict_file(self, wav_file_path):
        """
        Predict MOS score for a single WAV file.
        
        Args:
            wav_file_path (str): Path to WAV file
            
        Returns:
            dict: Dictionary containing all SIGMOS scores
        """
        # Validate input
        if not os.path.isfile(wav_file_path):
            raise FileNotFoundError(f"File not found: {wav_file_path}")
            
        if not wav_file_path.lower().endswith('.wav'):
            raise ValueError(f"File is not a WAV file: {wav_file_path}")
        
        # Load audio with librosa
        audio_data, sr = librosa.load(wav_file_path, sr=None, mono=False)
        if audio_data.ndim > 1:
            # Convert to mono by averaging all channels (standardized approach)
            audio_data = np.mean(audio_data, axis=0)
        
        # Make prediction
        try:
            result = self.model.run(audio=audio_data, sr=sr)
            
            # Return the full result dictionary
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to predict MOS for {wav_file_path}: {str(e)}")
    
    def predict_audio(self, audio_data, sample_rate):
        """
        Predict MOS score for audio data.
        
        Args:
            audio_data (numpy.ndarray): Audio samples
            sample_rate (int): Sample rate of audio
            
        Returns:
            dict: Dictionary containing all SIGMOS scores
        """
        # Ensure mono audio by averaging all channels (standardized approach)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)
        
        # Make prediction
        try:
            result = self.model.run(audio=audio_data, sr=sample_rate)
            
            # Return the full result dictionary
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to predict MOS for audio data: {str(e)}")
    
    def predict_batch(self, wav_file_paths):
        """
        Predict MOS scores for a batch of WAV files.
        
        Args:
            wav_file_paths (list): List of paths to WAV files
            
        Returns:
            dict: Dictionary mapping file paths to MOS scores
        """
        if not wav_file_paths:
            return {}
        
        # Process files one by one
        results = {}
        for wav_path in wav_file_paths:
            if not os.path.isfile(wav_path) or not wav_path.lower().endswith('.wav'):
                continue
                
            try:
                # Use the individual file prediction method
                results[wav_path] = self.predict_file(wav_path)
            except Exception as e:
                logger.error(f"Error processing file {wav_path}: {e}")
                continue
        
        return results
    
    def predict_from_paths(self, paths):
        """
        Predict MOS scores for files or directories.
        
        Args:
            paths (list): List of file or directory paths
            
        Returns:
            dict: Dictionary mapping file paths to MOS scores
        """
        # Collect all WAV files from the given paths
        all_wav_files = []
        for path in paths:
            all_wav_files.extend(get_wav_files_from_path(path))
        
        # Process files individually
        return self.predict_batch(all_wav_files)


# Singleton-style utility functions
def predict_mos(wav_file_path, gpu_id=0, config=None):
    """
    Predict MOS score for a single WAV file.
    
    Args:
        wav_file_path (str): Path to WAV file
        gpu_id (int): GPU ID to use
        config: Optional configuration object
        
    Returns:
        dict: Dictionary containing all SIGMOS scores
    """
    pipeline = SIGMOSPipeline(gpu_id=gpu_id, config=config)
    return pipeline.predict_file(wav_file_path)


def predict_audio_mos(audio_data, sample_rate, gpu_id=0, config=None):
    """
    Predict MOS score for audio data.
    
    Args:
        audio_data (numpy.ndarray): Audio samples
        sample_rate (int): Sample rate of audio
        gpu_id (int): GPU ID to use
        config: Optional configuration object
        
    Returns:
        dict: Dictionary containing all SIGMOS scores
    """
    pipeline = SIGMOSPipeline(gpu_id=gpu_id, config=config)
    return pipeline.predict_audio(audio_data, sample_rate)


def predict_batch_mos(wav_file_paths, gpu_id=0, config=None):
    """
    Predict MOS scores for a batch of WAV files.
    
    Args:
        wav_file_paths (list): List of paths to WAV files
        gpu_id (int): GPU ID to use
        config: Optional configuration object
        
    Returns:
        dict: Dictionary mapping file paths to MOS scores
    """
    pipeline = SIGMOSPipeline(gpu_id=gpu_id, config=config)
    
    # Check if any of the paths is a directory
    has_dir = False
    for path in wav_file_paths:
        if os.path.isdir(path):
            has_dir = True
            break
    
    if has_dir:
        return pipeline.predict_from_paths(wav_file_paths)
    else:
        return pipeline.predict_batch(wav_file_paths)
