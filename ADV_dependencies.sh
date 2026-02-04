#!/bin/bash

# Installation script for audio processing dependencies
# Make sure your virtual environment is activated before running this script

set -e  # Exit on error

echo "Installing dependencies in virtual environment..."

# Upgrade pip first (recommended)
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Core build utilities
echo "Installing core build utilities..."
python -m pip install Cython packaging

# Audio Processing
echo "Installing torchaudio..."
python -m pip install torchaudio

# NISQA dependency 
echo "Installing seaborn..."
python -m pip install seaborn==0.11.2

# ONNX and ONNX Runtime (GPU-enabled)
echo "Installing ONNX and ONNX Runtime..."
python -m pip install onnx==1.19.0 onnxruntime==1.23.2 onnxruntime-gpu

# Torch codec utilities
echo "Installing torchcodec..."
python -m pip install torchcodec

# Silero VAD
echo "Installing silero-vad..."
python -m pip install silero-vad

echo "All dependencies installed successfully!"
