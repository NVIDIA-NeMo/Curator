# Synthetic Data Generation Tutorials

Hands-on tutorials for generating synthetic data with NeMo Curator using Ray-based distributed processing.

## Documentation

For comprehensive documentation, refer to the [Synthetic Data Generation Guide](../../docs/curate-text/synthetic/index.md).

## Getting Started

### Prerequisites

- **NVIDIA API Key**: Obtain from [NVIDIA Build](https://build.nvidia.com/settings/api-keys)
- **NeMo Curator**: Installed with text extras (`pip install nemo-curator[text_cuda12]`)

### Setup

```bash
export NVIDIA_API_KEY="your-api-key-here"
```

## Available Tutorials

| Tutorial | Description | Difficulty |
|----------|-------------|------------|
| [Multilingual Q&A](synthetic_data_generation_example.py) | Generate Q&A pairs in multiple languages | Beginner |
| [Nemotron-CC High-Quality](nemotron_cc/nemotron_cc_sdg_high_quality_example_pipeline.py) | Advanced SDG for high-quality data (DiverseQA, Distill, ExtractKnowledge, KnowledgeList) | Advanced |
| [Nemotron-CC Low-Quality](nemotron_cc/nemotron_cc_sdg_low_quality_example_pipeline.py) | Improve low-quality data via Wikipedia-style paraphrasing | Advanced |

## Quick Examples

### Basic Multilingual Q&A

```bash
# Generate 20 synthetic Q&A pairs in multiple languages
python synthetic_data_generation_example.py --num-samples 20

# Customize languages and disable filtering
python synthetic_data_generation_example.py \
    --num-samples 50 \
    --languages English French German Spanish \
    --no-filter-languages
```

### Nemotron-CC Pipelines

```bash
# High-quality processing: Run any task (diverse_qa, distill, extract_knowledge, knowledge_list)
python nemotron_cc/nemotron_cc_sdg_high_quality_example_pipeline.py \
    --task diverse_qa \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --mock

# Low-quality processing: Wikipedia-style paraphrasing to improve text quality
python nemotron_cc/nemotron_cc_sdg_low_quality_example_pipeline.py \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --mock
```

### Using Real Data

```bash
# Process Parquet input files
python nemotron_cc/nemotron_cc_sdg_high_quality_example_pipeline.py \
    --task diverse_qa \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --input-parquet-path ./my_data/*.parquet \
    --output-path ./synthetic_output \
    --output-format parquet
```

## Command-Line Arguments

Refer to each script's `--help` output for the complete list of available arguments.

| Argument | Default | Description |
|----------|---------|-------------|
| `--api-key` | env var | NVIDIA API key |
| `--base-url` | NVIDIA API | Base URL for API endpoint |
| `--model-name` | meta/llama-3.3-70b-instruct | Model to use for generation |
| `--output-path` | ./synthetic_output | Output directory |
| `--max-concurrent-requests` | 3 | Concurrent API requests |
| `--temperature` | 0.9 (QA) / 0.5 (Nemotron-CC) | Sampling temperature |

## Example Output

### Multilingual Q&A

```json
{"text": "[EN] Question: What causes ocean tides? Answer: Ocean tides are primarily caused by the gravitational pull of the Moon and Sun on Earth's water bodies."}
{"text": "[FR] Question: Qu'est-ce que la photosynthèse? Answer: La photosynthèse est le processus par lequel les plantes convertissent la lumière du soleil en énergie."}
```

### Nemotron-CC

See the [Nemotron-CC documentation](../../docs/curate-text/synthetic/nemotron-cc/index.md) for output format details for each task type.

---

## Additional Resources

- [LLM Client Configuration](../../docs/curate-text/synthetic/llm-client.md)
- [Nemotron-CC Pipeline Documentation](../../docs/curate-text/synthetic/nemotron-cc/index.md)
- [Task Reference](../../docs/curate-text/synthetic/nemotron-cc/tasks.md)
