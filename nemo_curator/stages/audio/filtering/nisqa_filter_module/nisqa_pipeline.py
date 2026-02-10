#!/usr/bin/env python3
"""
NISQA Pipeline: A module for processing WAV files to get MOS predictions in pipelines.
This standalone module can be imported and used in other scripts without modifying
the original batch_mos_pred.py script.
"""

import os
import sys
import torch
import pandas as pd
import time
import copy
import threading
from pathlib import Path
from loguru import logger

# IMPORTANT: This module is intended to be self-contained within nemo_curator
# It dynamically adds the NISQA package to sys.path at runtime
# We assume the third_party directory structure is preserved relative to this file

# Model cache for storing initialized models by GPU ID
# Thread-safe access via _MODEL_CACHE_LOCK
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()

# Per-model prediction locks to ensure thread-safe inference
# Key: gpu_id, Value: threading.Lock()
_PREDICTION_LOCKS = {}
_PREDICTION_LOCKS_LOCK = threading.Lock()


def _get_prediction_lock(gpu_id: int) -> threading.Lock:
    """Get or create a prediction lock for a specific GPU."""
    global _PREDICTION_LOCKS, _PREDICTION_LOCKS_LOCK
    with _PREDICTION_LOCKS_LOCK:
        if gpu_id not in _PREDICTION_LOCKS:
            _PREDICTION_LOCKS[gpu_id] = threading.Lock()
        return _PREDICTION_LOCKS[gpu_id]


def build_nisqa_model(config=None, nisqa_base_path=None):
    """
    Build and return a NISQA model instance
    
    Args:
        config: Optional configuration object (can be dict or object with .get method)
        nisqa_base_path: Base path where NISQA third_party folder is located
    
    Returns:
        NISQA model instance
    """
    # Setup path to import NISQA third_party modules
    if nisqa_base_path is None:
        # Default to current directory if not provided
        nisqa_base_path = os.path.dirname(os.path.abspath(__file__))
    
    if nisqa_base_path not in sys.path:
        sys.path.insert(0, nisqa_base_path)
    
    # Try importing NISQA model - handled inside here to catch import errors if paths are wrong
    try:
        from third_party.NISQA.NISQA_model import nisqaModel
    except ImportError:
        # If relative import fails, try absolute from the module location
        try:
            sys.path.append(os.path.join(nisqa_base_path, "third_party"))
            from NISQA.NISQA_model import nisqaModel
        except ImportError as e:
            logger.error(f"Could not import NISQA model. Check if 'third_party/NISQA' exists at {nisqa_base_path}")
            raise e

    # Default NISQA checkpoint path - relative to the module
    default_nisqa_ckpt = os.path.join(nisqa_base_path, "third_party/NISQA/weights/nisqa.tar")
    
    # Build arguments for NISQA model
    args = {}
    args["mode"] = "predict_dir"
    args["deg"] = None
    
    # Use config if provided, otherwise use defaults
    # Handle both dictionary and object-like config
    def get_config_val(section, key, default):
        if config is None:
            return default
        if isinstance(config, dict):
            # Check nested dict
            val = config.get(section, {}).get(key)
            if val is not None: return val
            # Check flat key e.g. "nisqa_model_path"
            val = config.get(f"{section}_{key}")
            if val is not None: return val
            # Also check simplified flat key if section is part of name
            if key.startswith(f"{section}_"):
                 val = config.get(key)
                 if val is not None: return val
            return default
        if hasattr(config, 'get'): # Config object with get method
             return config.get(section, key, default)
        # Direct attribute access attempt (fallback)
        val = getattr(config, f"{section}_{key}", None)
        if val is not None:
             return val
        return default

    if config:
        # Check for model path in config, otherwise use default local path
        model_path_cfg = get_config_val('nisqa', 'model_path', None)
        if model_path_cfg and os.path.exists(model_path_cfg):
             args["pretrained_model"] = model_path_cfg
        else:
             args["pretrained_model"] = default_nisqa_ckpt
             
        args["num_workers"] = get_config_val('nisqa', 'num_workers', 0)
        args["bs"] = get_config_val('nisqa', 'batch_size', 1)
        args["ms_channel"] = get_config_val('nisqa', 'channel', None)
        args["disable_print"] = not get_config_val('general', 'verbose', False)
        args["skip_avg"] = get_config_val('nisqa', 'skip_avg', True)
        args["defer_dataloading"] = get_config_val('nisqa', 'defer_dataloading', True)
        args["use_gpu"] = get_config_val('nisqa', 'use_gpu', True)
    else:
        args["pretrained_model"] = default_nisqa_ckpt
        args["num_workers"] = 0
        args["bs"] = 1
        args["ms_channel"] = None
        args["disable_print"] = True
        args["skip_avg"] = True
        args["defer_dataloading"] = True
        args["use_gpu"] = True
    
    args["data_dir"] = None
    args["output_dir"] = None
    args["csv_file"] = None
    args["csv_deg"] = None
    args["tr_bs_val"] = args["bs"]
    args["tr_num_workers"] = args["num_workers"]

    # Debug print to confirm model path
    print(f"DEBUG: Loading NISQA model from {args['pretrained_model']}")
    return nisqaModel(args)


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


class NISQAPipeline:
    """
    A class for predicting MOS scores for audio files that can be used in pipelines.
    Thread-safe for multi-GPU parallel processing.
    """
    def __init__(self, gpu_id=0, config=None, base_path=None):
        """
        Initialize the MOS predictor.
        
        Args:
            gpu_id (int): GPU ID to use
            config: Optional configuration object
            base_path: Optional base path for resolving NISQA modules/weights
        """
        self.gpu_id = gpu_id
        self.config = config
        self.base_path = base_path or os.path.dirname(os.path.abspath(__file__))
        
        # Set device for this thread
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
        
        # Initialize model or get from cache (thread-safe)
        global _MODEL_CACHE, _MODEL_CACHE_LOCK
        
        with _MODEL_CACHE_LOCK:
            if self.gpu_id not in _MODEL_CACHE:
                # Ensure we're on the correct GPU before loading
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.gpu_id)
                _MODEL_CACHE[self.gpu_id] = build_nisqa_model(config, self.base_path)
            
            self.model = _MODEL_CACHE[self.gpu_id]
    
    def predict_file(self, wav_file_path):
        """
        Predict NISQA scores for a single WAV file.
        
        Thread-safe: Uses per-GPU prediction locks to ensure the underlying
        model is not accessed concurrently by multiple threads.
        
        Args:
            wav_file_path (str): Path to WAV file
            
        Returns:
            dict: Dictionary containing all NISQA scores (mos_pred, noi_pred, col_pred, dis_pred, loud_pred)
        """
        # Validate input
        if not os.path.isfile(wav_file_path):
            raise FileNotFoundError(f"File not found: {wav_file_path}")
            
        if not wav_file_path.lower().endswith('.wav'):
            raise ValueError(f"File is not a WAV file: {wav_file_path}")
        
        # Ensure we're on the correct GPU for this prediction
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
        
        # Get directory and filename
        dir_name = os.path.dirname(wav_file_path)
        file_name = os.path.basename(wav_file_path)
        
        # Use prediction lock to ensure thread-safe access to the model
        # The NISQA model (predict_v3) is NOT thread-safe
        prediction_lock = _get_prediction_lock(self.gpu_id)
        with prediction_lock:
            df = self.model.predict_v3(data_dir=dir_name, file_list=[file_name])
        
        if df.empty:
            raise RuntimeError(f"Failed to predict MOS for {wav_file_path}")
        
        # Return dictionary with all scores
        row = df.iloc[0]
        return {
            'mos_pred': row.get('mos_pred', 0.0),
            'noi_pred': row.get('noi_pred', 0.0),
            'col_pred': row.get('col_pred', 0.0),
            'dis_pred': row.get('dis_pred', 0.0),
            'loud_pred': row.get('loud_pred', 0.0),
        }
    
    def predict_batch(self, wav_file_paths):
        """
        Predict NISQA scores for a batch of WAV files.
        
        Args:
            wav_file_paths (list): List of paths to WAV files
            
        Returns:
            dict: Dictionary mapping file paths to NISQA score dictionaries
        """
        if not wav_file_paths:
            return {}
        
        # Process files one by one to avoid DataLoader collation issues
        results = {}
        for wav_path in wav_file_paths:
            if not os.path.isfile(wav_path) or not wav_path.lower().endswith('.wav'):
                continue
                
            try:
                # Use the individual file prediction method which returns full score dictionary
                score_dict = self.predict_file(wav_path)
                results[wav_path] = score_dict
            except Exception as e:
                print(f"Error processing file {wav_path}: {e}")
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
        
        # Process files individually instead of in batch
        results = {}
        for wav_file in all_wav_files:
            try:
                score_dict = self.predict_file(wav_file)
                results[wav_file] = score_dict
            except Exception as e:
                print(f"Error processing file {wav_file}: {e}")
                continue
                
        return results


# Singleton-style utility functions
def predict_mos(wav_file_path, gpu_id=0, config=None):
    """
    Predict NISQA scores for a single WAV file.
    
    Args:
        wav_file_path (str): Path to WAV file
        gpu_id (int): GPU ID to use
        config: Optional configuration object
        
    Returns:
        dict: Dictionary containing all NISQA scores
    """
    pipeline = NISQAPipeline(gpu_id=gpu_id, config=config)
    return pipeline.predict_file(wav_file_path)


def predict_batch_mos(wav_file_paths, gpu_id=0, config=None):
    """
    Predict NISQA scores for a batch of WAV files.
    
    Args:
        wav_file_paths (list): List of paths to WAV files
        gpu_id (int): GPU ID to use
        config: Optional configuration object
        
    Returns:
        dict: Dictionary mapping file paths to NISQA score dictionaries
    """
    pipeline = NISQAPipeline(gpu_id=gpu_id, config=config)
    
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
