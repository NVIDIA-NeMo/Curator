# Installation Troubleshooting

Common installation issues and solutions.

## Dependency Conflicts

### transformers Version Conflict (Audio)

**Error**: Version conflict with transformers package

**Solution**:
```bash
echo "transformers==4.55.2" > override.txt
uv pip install nemo-curator[audio_cuda12] --override override.txt
```

### torch/torchvision Mismatch

**Error**: `RuntimeError: version mismatch`

**Solution**:
```bash
uv pip install torch torchvision --force-reinstall
```

## Build Failures

### flash-attn Build Failure

**Error**: `fatal error: cuda_runtime.h: No such file or directory`

**Solution**:
```bash
sudo apt-get install build-essential ninja-build
uv pip install --no-build-isolation flash-attn<=2.8.3
```

### FastText Build Failure

**Error**: `error: command 'g++' failed`

**Solution**:
```bash
sudo apt-get install build-essential
uv pip install fasttext==0.9.3
```

### video_cuda12 Build Failure

**Error**: Various build errors with CUDA packages

**Solution**:
```bash
uv pip install --no-build-isolation nemo-curator[video_cuda12]
```

## Missing Dependencies

### `ModuleNotFoundError: cudf`

**Cause**: GPU deduplication not installed

**Solution**:
```bash
uv pip install nemo-curator[text_cuda12]
```

### `ImportError: flash_attn`

**Cause**: flash-attn not installed or build failed

**Solution**:
```bash
uv pip install --no-build-isolation flash-attn<=2.8.3
```

### `ImportError: nemo.collections.asr`

**Cause**: NeMo Toolkit ASR not installed

**Solution**:
```bash
echo "transformers==4.55.2" > override.txt
uv pip install nemo-curator[audio_cuda12] --override override.txt
```

### `ffmpeg: command not found`

**Cause**: FFmpeg not installed

**Solution**:
```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/docker/common/install_ffmpeg.sh -o install_ffmpeg.sh
sudo bash install_ffmpeg.sh
```

## GPU Issues

### `torch.cuda.is_available() = False`

**Cause**: CUDA not properly configured

**Diagnosis**:
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

**Solution**:
```bash
# Reinstall PyTorch with CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### RAPIDS CUDA Version Mismatch

**Error**: `cudf requires CUDA 12.x`

**Cause**: Wrong CUDA version

**Solution**: 
- RAPIDS requires CUDA 12.x
- Check with `nvidia-smi`
- If CUDA 11.x, use `_cpu` extras instead

### Out of GPU Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce batch size in stage configuration
2. Use `_cpu` extras for non-critical stages
3. Use streaming mode for large datasets

## Network/Download Issues

### NGC Model Download Fails

**Cause**: Authentication or network issues

**Solution**:
```bash
# For models requiring auth
export NGC_API_KEY=your_key

# Check connectivity
curl -I https://api.ngc.nvidia.com
```

### pip/uv Timeout

**Cause**: Slow network

**Solution**:
```bash
uv pip install --timeout 300 nemo-curator[all]
```

## Ray/Pipeline Issues

### `ModuleNotFoundError` in Ray Workers

**Error**: `ModuleNotFoundError: No module named 'my_module'` in StageWorker

**Cause**: Custom classes defined in external files can't be found by Ray workers (they're separate processes).

**Solution**: Define all custom filters, modifiers, and stages in the same file as your pipeline:

```python
# BAD - Ray workers can't find this
sys.path.insert(0, "/path/to/my/code")
from my_stages import MyFilter  # Fails in Ray workers!

# GOOD - Self-contained
class MyFilter(DocumentFilter):
    def score_document(self, text: str) -> float:
        return len(text)
    def keep_document(self, score: float) -> bool:
        return score > 100
```

### `AttributeError: property 'name' has no setter`

**Error**: `AttributeError: property 'name' of 'MyStage' object has no setter`

**Cause**: Parent class has `name` as a read-only property.

**Solution**: Don't set `self.name` in `__init__`. The name is auto-generated or override via class attribute.

### Ray Shared Memory Warning

**Warning**: `The object store is using /tmp instead of /dev/shm`

**Cause**: Docker container has insufficient shared memory.

**Solution**: Add `--shm-size` to docker run (use 30%+ of RAM):

```bash
docker run --rm --shm-size=8g \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/nemo-curator:25.09 \
  python my_pipeline.py
```

## Docker Issues

### NGC Image Not Found

**Error**: `Error response from daemon: nvcr.io/nvidia/nemo-curator:latest: not found`

**Cause**: The `:latest` tag doesn't always exist on NGC.

**Solution**: Use a specific version tag:

```bash
# Check available tags at: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator
docker pull nvcr.io/nvidia/nemo-curator:25.09
```

## Docker Alternative

If native installation fails, use Docker:

```bash
docker pull nvcr.io/nvidia/nemo-curator:25.09
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo-curator:25.09
```

## Verification

After fixing issues, verify installation:

```bash
python .cursor/skills/setup/scripts/verify_installation.py --all
```
