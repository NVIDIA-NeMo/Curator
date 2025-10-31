# Synthetic Data Generation Tutorials

Hands-on tutorials for generating synthetic data with NeMo Curator. Complete working examples with detailed explanations.


## Getting Started

### Prerequisites

To run these tutorials, you'll need an NVIDIA API key. You can obtain one from:
- **NVIDIA Build**: https://build.nvidia.com/settings/api-keys

### Setup

Set your API key as an environment variable:

```bash
export NVIDIA_API_KEY="your-api-key-here"
```

Alternatively, you can pass it directly using the `--api-key` argument when running the examples.

### Quick Example

```bash
# Generate 20 synthetic Q&A pairs in multiple languages
python synthetic_data_generation_example.py --num-samples 20
```


## Available Tutorials

| Tutorial | Description | Files |
|----------|-------------|-------|
| **[Multilingual Q&A Generation](synthetic_data_generation_example.py)** | Generate synthetic Q&A pairs in multiple languages using LLMs | `synthetic_data_generation_example.py` |

