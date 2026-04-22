# Nemotron-Climb Data Curation

[CLustering-based Iterative Data Mixture Bootstrapping (Nemotron-CLIMB)](https://arxiv.org/abs/2504.13161) is an automated framework that discovers, evaluates, and refines data mixtures in a pre-training setting. Specifically, Nemotron-CLIMB embeds and clusters large-scale datasets in a semantic space and then iteratively searches for optimal mixtures using a smaller proxy model and a predictor.

This tutorial uses NeMo Curator to implement the data curation recipe used to create the [Nemotron-CLIMB dataset](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix). At a high level, it follows these steps:

1. Compute text embeddings per document
2. Cluster the documents with K-Means
3. Use FastText classification models to remove low-quality clusters, then merge similar clusters according to a Euclidean distance threshold
4. Tokenize and write to `.bin` and `.idx` files
5. Use a Dirichlet distribution to generate data mixtures
6. Train a proxy model for each data mixture using Megatron-LM
7. Benchmark each proxy model using LM Evaluation Harness
8. Fit a LightGBM predictor on the benchmark results and use the predictor to generate an optimal data mixture for full-scale LLM training

For reference, the [Nemotron-CLIMB dataset](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) was produced using this pipeline on multi-terabyte pretraining corpora. See the [Nemotron-CLIMB paper](https://arxiv.org/pdf/2504.13161) for full experimental details, ablations, and results.

## Step 0: Requirements

The following Python libraries are needed to run this tutorial:

- `nemo-curator`
- `xformers`
- `matplotlib`
- `megatron-core`
- `lm-eval`
- `lightgbm`

NeMo Curator should be installed with the `text_cuda12` extra. See the installation options on the [Text Quickstart](https://docs.nvidia.com/nemo/curator/get-started/text) for more information. Installation commands for the other libraries above are included within their respective sections.

To run NeMo Curator, the following system requirements are needed:

- Ubuntu 22.04/20.04
- NVIDIA GPU with:
  - Volta™ or higher (compute capability 7.0+)
  - CUDA 12.x

Refer to the [Megatron Core Installation documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html) for information about the system requirements for running Megatron-LM, which is used in step 6 of this tutorial.

## Step 1: Compute Embeddings

Compute text embeddings on the dataset with:

```bash
# run `uv pip install xformers` to use NovaSearch/stella_en_400M_v5 model

python 1_embed.py \
    --input-path /path/to/input/data/dir \
    --input-filetype "jsonl" \
    --output-path /path/to/computed_embeddings \
    --text-field "text" \
    --id-field "climb_id" \
    --use-sentence-transformer
```

At least 1 GPU is required to run this step. Use the `--num-cpus` and `--num-gpus` arguments as desired to control the number of CPUs and GPUs used by the Ray client. The script uses all available resources by default.

The script uses the [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) embedding model by default:

- The [xFormers](https://github.com/facebookresearch/xformers) library is required to run the [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) model and should be installed via `uv pip install xformers` before running the script.
- The [SentenceTransformers](https://huggingface.co/sentence-transformers) library is used via the `use-sentence-transformer` flag to enhance performance. It is already included as a dependency in Curator and does not need to be manually installed.

Some of the default parameters in the script include:

- Use `--model_inference_batch_size 1024` to create digestible batch sizes for the model forward pass. Adjust as necessary; decrease the size to address memory issues and increase the size to improve performance.
- Use [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5)'s `max_length` of 512 via the `--max-seq-length` argument
- Use `--padding-side "right"` and `--embedding_pooling="mean_pooling"` defaults as appropriate for the [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) model
- Use `--transformers-init-kwargs "{'trust_remote_code': true}"` as required to load the [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) model

See script for full list of parameters.

## Step 2: K-Means Clustering

Run K-Means clustering on the computed embeddings:

```bash
python 2_cluster.py \
    --input-path /path/to/computed_embeddings \
    --output-path /path/to/clusters \
    --text-field "text" \
    --id-field "climb_id" \
    --embedding-dim 3072 \
    --centroids-path /path/to/centroids
```

At least 1 GPU is required to run this step. Use the `--num-cpus` and `--num-gpus` arguments as desired to control the number of CPUs and GPUs used by the Ray client. The script uses all available resources by default.

The script uses `--n-clusters 1000` as the default. Set the `--embedding-dim` to 2-3x the embedding model's dimension to avoid GPU out-of-memory issues (the default dimension for [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) is 1024, so set `--embedding-dim 3072`).

The `--id-field` generated from step 1 is required. See script for full list of K-Means parameters.

## Step 3: Cluster Pruning

Use a FastText model to prune the created clusters:

```bash
FASTTEXT_MODEL_PATHS=(
    /path/to/best_model_advertisement.bin
    /path/to/best_model_cultural_value.bin
    /path/to/best_model_educational_value.bin
    /path/to/best_model_informational_value.bin
    /path/to/best_model_quality.bin
)
FASTTEXT_SCORE_FIELDS=(
    advertisement_score
    cultural_value_score
    educational_value_score
    informational_value_score
    quality_score
)
python 3_prune.py \
    --input-path /path/to/clusters \
    --output-path /path/to/pruned_clusters \
    --fasttext-model-paths ${FASTTEXT_MODEL_PATHS[@]} \
    --score-fields ${FASTTEXT_SCORE_FIELDS[@]} \
    --text-field "text" \
    --centroids-path /path/to/centroids \
    --merge-threshold 1.5
```

No GPUs are needed to run this step, so by default the script sets `--num-gpus 0`. Use the `--num-cpus` argument as desired to control the number of CPUs used by the Ray client; by default, all are used. However, because each FastText model is large, CPU out-of-memory errors may occur due to overhead between stage workers. Try decreasing the number of CPUs if needed.

There are 5 FastText quality models that can be used for this step. Each can be pre-downloaded from Hugging Face:

TODO: Update links when the models are published

- [best_model_advertisement.bin](https://huggingface.co/nvidia)
- [best_model_cultural_value.bin](https://huggingface.co/nvidia)
- [best_model_educational_value.bin](https://huggingface.co/nvidia)
- [best_model_informational_value.bin](https://huggingface.co/nvidia)
- [best_model_quality.bin](https://huggingface.co/nvidia)

Users may opt to run the script with all 5 models as demonstrated above, or a subset of the models. For each path in `--fasttext-model-paths`, a unique score field must be set via the `--score-fields` argument.

After the FastText scores are computed, clusters with an average score less than `--pruning-threshold 1.0` are removed. Finally, remaining clusters with a Euclidean distance closer than `--merge-threshold 1.5` are combined with each other.

## Step 4: Convert to Tokenized Files

Generate `.bin` and `.idx` files for each cluster:

```bash
python 4_tokenize.py \
    --input-path /path/to/pruned_clusters \
    --output-path /path/to/domains \
    --hf-token "hf_XXX" \
    --append-eod
```

By default, the script tokenizes the text using [https://huggingface.co/meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b), which is a gated model. The user can request access to it on Hugging Face and pass the `--hf-token` argument as demonstrated above.

The pipeline generates pairs of files with identical names: one with a `.bin` extension and another with a `.idx` extension. This ensures that the input data files are compatible with Megatron-LM, which is used in step 6 to train the proxy models. Megatron-LM refers to the filename without the file extension as the "file prefix."

## Step 5: Generate Training Data Mixtures

Generate a mixture of data ratios to be used for training the proxy models:

```bash
python 5_mixture.py \
    --input-path /path/to/domains \
    --output-path /path/to/mixtures \
    --num-mixtures 64
```

Using the `.bin` files generated by the previous step, the script calculates the token distribution across the files. It initializes a Dirichlet distribution based on each cluster's token count and sample configurations. The value of `--num-mixtures` mixtures are generated.

## Step 6: Train Proxy Models

TODO: Add instructions for using a base model checkpoint. This includes using the base proxy models released by NVIDIA or other open source models. Starting from scratch is a reasonable option as well, so long as the user spends time exploring the best setup for their needs to ensure that the models provide useful signals for comparison against each other.

Kick off a Megatron training job with a specified data mixture:

```bash
bash 6_train.sh \
    /path/to/Megatron-LM/pretrain_gpt.py \
    /path/to/mixtures/n1.sh \
    /path/to/megatron_exp/n1 \
    /path/to/Llama-2-7b/tokenizer.model
```

The above script requires Megatron-LM to be installed. System requirements and installation instructions can be found on the [Megatron Core Installation documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html) page. If using Docker, make sure to obtain an NGC API key from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/). Clone the Megatron-LM repository so that the [pretrain_gpt.py](https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py) can be obtained directly.

For reference, the tutorial was developed and tested by following these steps:

- `docker run --runtime nvidia --gpus all --ipc=host --shm-size=16g -it --rm -v /path/to/data:/path/to/data -v /path/to/Curator:/path/to/Curator nvcr.io/nvidia/nemo:26.02.nemotron_3_super`
- The Python script `/opt/Megatron-Bridge/3rdparty/Megatron-LM/pretrain_gpt.py` is already included inside the container and can be used as is.
- `bash /path/to/Curator/tutorials/text/nemotron-climb-data-curation/6_train.sh ...`

A proxy model can be trained for each of the data ratios generated by step 5. The above snippet trains a single proxy model using the first mixture. The script looks for 4 inputs in order:

- `PRETRAIN_GPT_PATH` (`/path/to/Megatron-LM/pretrain_gpt.py`): The path to Megatron-LM's [pretrain_gpt.py](https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py) script.
- `MIXTURE_SCRIPT` (`/path/to/mixtures/n1.sh`): The path to any of the mixture files generated by step 5.
- `WORK_PATH` (`/path/to/megatron_exp/n1`): The path to be used for checkpoints, caching, etc. The work path should be unique per training job.
- `TOKENIZER_MODEL` (`/path/to/Llama-2-7b/tokenizer.model`): The path to the tokenizer used in step 4.

If needed, download the tokenizer file with:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="meta-llama/Llama-2-7b",
    filename="tokenizer.model",
    token="hf_XXX",
)

print(path)
```

Since the goal is to train proxy models to compare against each other, the script triggers a timeout after roughly 2 hours via `--exit-duration-in-mins 110`. If 64 proxy models are trained, this amounts to roughly 120 hours spent training. Set a shorter runtime depending on available hardware and compute budgets.

See `6_train.sh` script for full list of parameters used during training and modify as desired. Tips and considerations are included in the comments of the script. Refer to the [Megatron Core documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/training-examples.html) for more information.

By default, all available GPUs are used for training. The script was developed and tested on a single node; modifications are needed to adapt it for multi-node runs.

## Step 7: Evaluate Proxy Models

Evaluate the proxy models using [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main):

```bash
bash 7_evaluate.sh \
    /path/to/Megatron-LM \
    /path/to/megatron_exp \
    /path/to/lm_eval_results \
    /path/to/Llama-2-7b/tokenizer.model \
```

LM Evaluation Harness can be installed with:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

The script requires Megatron to be installed. Here is an example for how to set up and run the evaluation step:

- `docker run --runtime nvidia --gpus all --ipc=host --shm-size=16g -it --rm -v /path/to/data:/path/to/data -v /path/to/Curator:/path/to/Curator nvcr.io/nvidia/nemo:26.02.nemotron_3_super`
- Install `lm-eval` as instructed above.
- `bash /path/to/Curator/tutorials/text/nemotron-climb-data-curation/7_evaluate.sh ...`

The above script looks for 4 inputs in order:

- `MEGATRON_PATH` (`/path/to/Megatron-LM`): The path to the Megatron-LM directory.
- `BASE_CKPT_DIR` (`/path/to/megatron_exp`): The path to the trained proxy models from step 6, with subdirectories `/path/to/megatron_exp/n1`, `/path/to/megatron_exp/n2`, etc. per model.
- `RESULTS_DIR` (`/path/to/lm_eval_results`): The path to save the benchmarking results.
- `TOKENIZER_MODEL` (`/path/to/Llama-2-7b/tokenizer.model`): The path to the tokenizer used in steps 4 and 6.

The script evalutes each proxy model on the [ARC-Easy](https://arxiv.org/abs/1803.05457), [HellaSwag](https://arxiv.org/abs/1905.07830), and [PIQA](https://arxiv.org/abs/1911.11641) benchmarks.

By default, all available GPUs are used for benchmarking. Make sure the `model_args` used in `7_evaluate.sh` match those used in `6_train.sh` (e.g., `tokenizer_type=Llama2Tokenizer` and `seq_length=1024`).

## Step 8: Fit Predictor

Fit a [LightGBM](https://lightgbm.readthedocs.io/en/stable/) predictor on the results with:

```bash
# run `uv pip install lightgbm` if not already installed

python 8_predict.py \
    --input-path /path/to/lm_eval_results \
    --domains-path /path/to/domains \
    --mixtures-path /path/to/mixtures \
    --output-path /path/to/predict_results \
    --metric "valid_avg" \
    --num-mixtures 1
```

The script uses `lightgbm` which can each be installed via `uv pip install lightgbm`. It requires several inputs:

- `--input-path`: The output `lm_eval_results` directory from step 7
- `--domains-path`: The output `domains` directory from step 4 containing the `.bin` and `.idx` files
- `--mixtures-path`: The output `mixtures` directory from step 5 containing the data mixtures
- `--output-path`: Path to write the output mixture(s)

The script fits a LightGBM predictor by using the data mixtures as the features and the average benchmark score (via `--metric "valid_avg"`) as the target. It then outputs `--num-mixtures 1` data ratios which represent the optimal data mixture(s) to use for LLM training.

From here, the user may opt to use `num_mixtures > 1` and repeat steps 6 and 7 with the newly generated data mixtures. Then, the new benchmarks can be combined with the existing benchmarks to rerun step 8 and fit a predictor with all benchmarking data. The goal is to iterate upon steps 6, 7, and 8 until a single optimal data mixture is produced to be used for full-scale LLM training.

## Conclusion

To follow the [Nemotron-CLIMB paper](https://arxiv.org/pdf/2504.13161) exactly:

- Run steps 1-7, generating 64 data mixtures in step 5 and training a single proxy model per data mixture (64 models total).
- Run step 8 to fit a predictor on the benchmarks from the 64 models. Generate 32 more mixtures.
- Re-run steps 6 and 7 with the 32 data mixtures.
- Re-run step 8 to fit a predictor using current and past benchmarks (64 + 32 = 96 total). Generate 16 more mixtures.
- Re-run steps 6 and 7 with the 16 data mixtures.
- Re-run step 8 to fit a predictor using current and past benchmarks (96 + 16 = 112 total). Generate 1 more mixture.
- Use the final data mixture for full-scale LLM training.

By iteratively refining the data mixture across three rounds (64 -> 32 -> 16 -> 1), the predictor is trained on an increasingly rich set of mixture-performance pairs, resulting in a more accurate estimate of the optimal data mixture. The final mixture can then be used to train a full-scale LLM with a data composition that is empirically grounded in downstream benchmark performance.
