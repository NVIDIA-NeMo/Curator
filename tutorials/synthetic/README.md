# Synthetic Data Generation Tutorials

Hands-on tutorials for generating synthetic data with NeMo Curator using Ray-based distributed processing.

## Documentation

For comprehensive documentation, refer to the [Synthetic Data Generation Guide](../../docs/curate-text/synthetic/index.md).

## Getting Started

### Prerequisites

- **NVIDIA API Key**: To use public Nvidia's endpoint, obtain Nvidia API key from [NVIDIA Build](https://build.nvidia.com/settings/api-keys).
```bash
export NVIDIA_API_KEY="your-api-key-here"
```
- **NeMo Curator**: Installed with text extras (`uv pip install "nemo-curator[text_cuda12]"`)


## SDG Backends

NeMo Curator supports two backends for synthetic data generation:

- **OpenAI Client-based**: A lightweight approach where you manually craft prompt templates and call any OpenAI-compatible REST endpoint via `AsyncOpenAIClient` + `GenerationConfig`. Best for custom pipelines where you want full control over prompt logic and request parameters.
- **NeMo Data Designer (NDD)**: A declarative, high-level framework where you define a data schema (structured samplers, Jinja expressions, LLM-generated columns) and NDD handles prompt rendering, batching, and concurrency automatically. Best for structured dataset generation with minimal boilerplate.

The tutorials below cover both backends across a range of use cases.

## Available Tutorials

### OpenAI Client-based

These tutorials use NeMo Curator's `AsyncOpenAIClient` paired with a `GenerationConfig` to call any OpenAI-compatible REST endpoint (e.g. NVIDIA NIM, self-hosted vLLM, or OpenAI itself).
You write the prompt template and control generation parameters (temperature, top-p, token limits) directly in Python.
This approach is lightweight and portable — no additional frameworks are required beyond NeMo Curator — making it a good starting point for custom pipelines where you want full control over the request logic.

| Tutorial | Description | Difficulty |
|----------|-------------|------------|
| [Multilingual Q&A](synthetic_data_generation_example.py) | Generate Q&A pairs in multiple languages | Beginner |
| [Nemotron-CC High-Quality](nemotron_cc/nemotron_cc_sdg_high_quality_example_pipeline.py) | Advanced SDG for high-quality data (DiverseQA, Distill, ExtractKnowledge, KnowledgeList) | Advanced |
| [Nemotron-CC Low-Quality](nemotron_cc/nemotron_cc_sdg_low_quality_example_pipeline.py) | Improve low-quality data via Wikipedia-style paraphrasing | Advanced |

### NeMo Data Designer (NDD)-based

[NeMo Data Designer](https://developer.nvidia.com/nemo-data-designer) is NVIDIA's high-level synthetic data generation framework.
Instead of hand-crafting prompt strings and managing API calls, you declare your data schema — samplers for structured fields (names, dates, UUIDs), Jinja-style expression columns, and LLM-generated text columns — using a `DataDesignerConfigBuilder`.
NDD then orchestrates prompt rendering, batching, and concurrency automatically via its `ModelConfig` / `ModelProvider` / `ChatCompletionInferenceParams` API.

Key advantages over the OpenAI Client-based approach:
- **Structured column generation**: mix faker-based samplers, datetime ranges, UUID generators, and LLM-generated text in a single config.
- **Local or remote inference**: point NDD at a local `InferenceServer` (Ray Serve + vLLM) or any remote OpenAI-compatible endpoint without changing pipeline code.
- **Declarative config**: pipelines can be defined in YAML (`--data-designer-config-file`) for easy reproducibility and sharing.

| Tutorial | Description | Difficulty |
|----------|-------------|------------|
| [NDD Medical Notes](nemo_data_designer/ndd_data_generation_example.py) | Generate synthetic medical notes from symptom seed data; supports local vLLM server or remote provider | Intermediate |
| [NDD Nemotron-CC High-Quality](nemotron_cc/nemo_data_designer/nemotron_cc_sdg_high_quality_example_pipeline.py) | NDD-backed SDG for high-quality data (DiverseQA, Distill, ExtractKnowledge, KnowledgeList) | Advanced |
| [NDD Nemotron-CC Low-Quality](nemotron_cc/nemo_data_designer/nemotron_cc_sdg_low_quality_example_pipeline.py) | NDD-backed Wikipedia-style paraphrasing to improve low-quality data | Advanced |

## Quick Examples

### OpenAI Client-based

#### Multilingual Q&A

```bash
# Generate 20 synthetic Q&A pairs in multiple languages
python synthetic_data_generation_example.py --num-samples 10

# Customize languages and disable filtering
python synthetic_data_generation_example.py \
    --num-samples 10 \
    --languages English French German Spanish \
    --no-filter-languages
```

#### Nemotron-CC Pipelines

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

#### Using Real Data

```bash
# Process Parquet input files
python nemotron_cc/nemotron_cc_sdg_high_quality_example_pipeline.py \
    --task diverse_qa \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --input-parquet-path ./my_data/*.parquet \
    --output-path ./synthetic_output \
    --output-format parquet
```

### NeMo Data Designer (NDD)-based

#### Medical Notes Generation

```bash
# Remote NVIDIA NIM API
python nemo_data_designer/ndd_data_generation_example.py \
    --provider nvidia \
    --model meta/llama-3.3-70b-instruct
```

#### Nemotron-CC Pipelines

```bash
# High-quality processing: Run any task (diverse_qa, distill, extract_knowledge, knowledge_list)
python nemotron_cc/nemo_data_designer/nemotron_cc_sdg_high_quality_example_pipeline.py \
    --task diverse_qa \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --mock

# Low-quality processing: Wikipedia-style paraphrasing to improve text quality
python nemotron_cc/nemo_data_designer/nemotron_cc_sdg_low_quality_example_pipeline.py \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --mock
```

#### Using Real Data

```bash
# Process Parquet input files
python nemotron_cc/nemo_data_designer/nemotron_cc_sdg_high_quality_example_pipeline.py \
    --task diverse_qa \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --input-parquet-path ./my_data/*.parquet \
    --output-path ./synthetic_output \
    --output-format parquet
```

---

## Additional Resources

- [LLM Client Configuration](../../docs/curate-text/synthetic/llm-client.md)
- [Nemotron-CC Pipeline Documentation](../../docs/curate-text/synthetic/nemotron-cc/index.md)
- [Task Reference](../../docs/curate-text/synthetic/nemotron-cc/tasks.md)
