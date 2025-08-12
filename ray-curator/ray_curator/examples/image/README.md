# Image Processing Examples - Ray Curator

This directory contains examples for image processing pipelines using the Ray Curator framework. These examples demonstrate various image processing capabilities including VLM-based caption generation, quality scoring, and filtering.

## VLM Caption Generation Pipeline

The VLM (Vision Language Model) caption generation pipeline allows you to generate descriptive captions for images using NVIDIA's VLM models through both local NIM endpoints and NVIDIA NGC cloud API.

### Features

- **Multiple Input Sources**: Support for local images, URL lists, and webdataset formats
- **Dual Endpoint Support**: Works with both local NIM endpoints and NVIDIA NGC cloud API
- **Async Processing**: Concurrent caption generation for improved throughput
- **Flexible Output**: Save results as JSON or webdataset format
- **NVIDIA_API_KEY Support**: Full support for NGC API authentication
- **Model Flexibility**: Support for any VLM model compatible with OpenAI specification
- **Customizable Prompts**: User-defined prompts for caption generation

### Prerequisites

1. **Model Access**: Choose one of the following:
   - **Local NIM Endpoint**: Running VLM NIM container locally
   - **NVIDIA NGC API**: Account with access to VLM models

2. **Installation**: Set up Ray Curator with dependencies

#### Install from PyProject.toml
```bash
# Create and activate virtual environment
python -m venv ray-curator-env
source ray-curator-env/bin/activate  # On Windows: ray-curator-env\Scripts\activate

# Install Ray Curator with all dependencies
cd /path/to/ray-curator
pip install -e .

# Install additional dependencies for image processing (if not already available)
pip install pillow aiohttp aiofiles

# Or install with specific optional dependencies
pip install -e .[text,video]  # For text and video processing
```

> **📋 Note**: Some image processing dependencies (pillow, aiohttp, aiofiles) are not explicitly declared in pyproject.toml but are required for VLM captioning.

3. **Ray Cluster Setup**: Start Ray cluster for distributed processing
```bash
# Start Ray cluster with dashboard (recommended for monitoring)
RAY_MAX_LIMIT_FROM_API_SERVER=50000 ray start \
    --include-dashboard=True \
    --dashboard-host=0.0.0.0 \
    --port=8265 \
    --dashboard-port=8266 \
    --head

# Access Ray Dashboard at: http://localhost:8266
```

### Sample Data

The `data/` folder contains sample images and URL lists for testing:
- `data/sample_images/` - Sample images including NGC performance charts
- `data/sample_urls.txt` - URL list for testing the `--source-type urls` option


#### Using Local NIM Endpoint
```bash
python vlm_caption_generation.py \
    --input-path /full/path/to/data/sample_images \
    --output-path /full/path/to/captions.json \
    --model-name nvidia/llama-3.1-nemotron-nano-vl-8b-v1 \
    --enable-async \
    --verbose
```

#### Using NVIDIA NGC Cloud API
```bash
# Set your API key
export NVIDIA_API_KEY="your_ngc_api_key"

python vlm_caption_generation.py \
    --input-path /full/path/to/data/sample_images \
    --output-path /full/path/to/captions.json \
    --use-ngc \
    --model-name nvidia/llama-3.1-nemotron-nano-vl-8b-v1 \
    --enable-async \
    --verbose
```

### API Endpoint Configuration

#### Local NIM Endpoint (Default)
- **Default URL**: `http://localhost:8000/v1`
- **API Key**: Not required
- **Setup**: Run NIM container locally

```bash
python vlm_caption_generation.py \
    --input-path /full/path/to/my_images \
    --output-path /full/path/to/captions.json \
    --nim-endpoint http://localhost:8000/v1
```

#### NVIDIA NGC Cloud API
- **URL**: `https://integrate.api.nvidia.com/v1`
- **API Key**: Required (NVIDIA_API_KEY)
- **Setup**: NGC account with model access

```bash
export NVIDIA_API_KEY="your_api_key_here"
python vlm_caption_generation.py \
    --input-path /full/path/to/my_images \
    --output-path /full/path/to/captions.json \
    --use-ngc
```

### Input Source Types

#### 1. Local Images
Process images from a local directory or single file:
```bash
python vlm_caption_generation.py \
    --input-path /path/to/images \
    --source-type local \
    --output-path captions.json
```

#### 2. URL List
Process images from URLs listed in a text file:
```bash
python vlm_caption_generation.py \
    --input-path /full/path/to/data/sample_urls.txt \
    --source-type urls \
    --output-path /full/path/to/captions.json
```

The `urls.txt` file should contain one URL per line:
```
https://example.com/image1.jpg
https://example.com/image2.png
https://example.com/image3.jpg
```

#### 3. Webdataset
Process images from webdataset tar files:
```bash
python vlm_caption_generation.py \
    --input-path /path/to/webdataset_directory/ \
    --source-type webdataset \
    --output-path captions.json
```

### Output Formats

#### JSON Output
Save results as a JSON file with image metadata and captions:
```bash
--output-format json --output-path captions.json
```

Example JSON structure:
```json
[
  {
    "image_id": "image001",
    "image_path": "/path/to/image001.jpg",
    "caption": "A detailed description of the image content...",
    "metadata": {
      "caption": "A detailed description of the image content..."
    }
  }
]
```

#### Webdataset Output
Save results as a webdataset with images and captions:
```bash
--output-format webdataset --output-path /path/to/output_dataset
```

### VLM Model Support

The pipeline supports any VLM model that is compatible with the OpenAI vision API specification. Available models include:

#### NVIDIA Models (via NGC or NIM)
- `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` (default)
- `mistralai/mistral-small-3.2-24b-instruct-2506`

#### Custom Models
Any model that supports the OpenAI vision API format with:
- Chat completions endpoint
- Image + text input messages
- Base64 encoded images

### Performance Optimization

#### Async Processing
Enable concurrent processing for better throughput:
```bash
--enable-async \
--max-concurrent-requests 10 \
--max-retries 3 \
--retry-delay 2.0
```

#### Batch Processing
Control batch sizes for memory management:
```bash
--reader-batch-size 32 \
--image-limit 1000
```

### Advanced Configuration

#### Custom Prompts
Specify custom prompts for caption generation:
```bash
--prompt "Describe this image focusing on the main objects and their relationships."
```

#### Model Parameters
Control VLM generation parameters:
```bash
--max-tokens 200 \
--temperature 0.7
```

#### Endpoint Configuration
Configure custom endpoints:
```bash
--nim-endpoint http://your-nim-server:8000/v1 \
--model-name your-vlm-model-name \
--timeout 300
```

## Command Line Reference

### Required Arguments
- `--input-path`: Path to input (file, directory, or URL list file)
- `--output-path`: Path for output (JSON file or webdataset directory)

### Input/Output Options
- `--source-type`: Input source type (`local`, `urls`, `webdataset`)
- `--output-format`: Output format (`json`, `webdataset`)
- `--image-limit`: Limit number of images to process (-1 for no limit)
- `--reader-batch-size`: Number of images per batch for reading

### API Configuration
- `--use-ngc`: Use NVIDIA NGC cloud API instead of local NIM
- `--api-key`: API key for NGC endpoint (can also use NVIDIA_API_KEY env var)
- `--nim-endpoint`: Custom NIM endpoint URL (ignored if --use-ngc is set)

### VLM Model Configuration
- `--model-name`: VLM model name (default: `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`)
- `--prompt`: Prompt for caption generation
- `--max-tokens`: Maximum tokens for caption generation
- `--temperature`: Temperature for caption generation

### Async Processing Options
- `--enable-async`: Enable async processing for faster caption generation
- `--max-concurrent-requests`: Maximum concurrent requests for async processing
- `--timeout`: Timeout for API requests in seconds
- `--max-retries`: Maximum number of retry attempts
- `--retry-delay`: Base delay between retries in seconds

### Output Configuration (Webdataset)
- `--samples-per-shard`: Number of samples per shard in output webdataset
- `--max-shards`: Maximum number of shards for output webdataset

### General Options
- `--verbose`: Enable verbose logging

## Examples


### Example 1: Local Images with NGC API
```bash
export NVIDIA_API_KEY="your_ngc_api_key"

python vlm_caption_generation.py \
    --input-path /full/path/to/my_images \
    --source-type local \
    --output-path /full/path/to/captions.json \
    --use-ngc \
    --model-name nvidia/llama-3.1-nemotron-nano-vl-8b-v1 \
    --prompt "Provide a detailed description of this image, including objects, people, and scene context." \
    --enable-async \
    --max-concurrent-requests 8 \
    --verbose
```

### Example 2: URL List with Local NIM
```bash

python vlm_caption_generation.py \
    --input-path /full/path/to/image_urls.txt \
    --source-type urls \
    --output-path /full/path/to/captioned_dataset \
    --output-format webdataset \
    --nim-endpoint http://localhost:8000/v1 \
    --image-limit 500 \
    --samples-per-shard 100 \
    --enable-async
```

### Example 3: Webdataset Processing with Custom Model
```bash

python vlm_caption_generation.py \
    --input-path /full/path/to/input_webdataset \
    --source-type webdataset \
    --output-path /full/path/to/captions.json \
    --use-ngc \
    --api-key "your_api_key" \
    --model-name nvidia/vila \
    --reader-batch-size 16 \
    --max-tokens 100 \
    --temperature 0.5
```
