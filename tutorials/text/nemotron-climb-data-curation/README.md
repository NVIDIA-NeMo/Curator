# Nemotron-Climb Data Curation

TODO: Description here

## Step 0: Requirements

- `nemo-curator`
- `lightgbm`
- `seaborn`
- `xformers`

## Step 1: Compute Embeddings

Compute text embeddings on the dataset with:

```bash
python 1_compute_embeddings.py \
    --input-path /path/to/input/data/dir \
    --input-file-type "jsonl" \
    --output-path /path/to/computed_embeddings \
    --output-file-type "jsonl" \
    --text-field "text" \
    --id-field "climb_id" \
    --use-sentence-transformer \
    --autocast \
    --sort-by-length
```

At least 1 GPU is required to run this step. Use the `--num-cpus` and `--num-gpus` arguments as desired to control the number of CPUs and GPUs used by the Ray client. The script uses all available resources by default.

If the dataset already contains a unique ID field, feel free to omit the `--id-field` argument.

The script uses the [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) embedding model by default. The [SentenceTransformers](https://huggingface.co/sentence-transformers) library is used via the `--use-sentence-transformer` flag to enhance performance. The `--autocast` flag leverages mixed-precision inference, which can speed up embedding generation and reduce memory usage on GPUs. The `--sort-by-length` flag is a Curator-specific function which sorts the input data by the length of the input tokens; sorting is encouraged to improve performance.

Some of the default parameters in the script include:

- Use `--model_inference_batch_size 1024` to create digestible batch sizes for the model forward pass. Adjust as necessary; decrease the size to address memory issues and increase the size to improve performance.
- Use [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5)'s `max_length` of 512 via the `--max-seq-length` argument
- Use `--padding-side "right"` and `--embedding_pooling="mean_pooling"` defaults as appropriate for the [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) model
- Use `--transformers-init-kwargs "{'trust_remote_code': true}"` as required to load the [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) model

See script for full list of parameters.

## Step 2: K-Means Clustering

Run K-Means clustering on the computed embeddings:

```bash
python 2_clustering.py \
    --input-path /path/to/computed_embeddings \
    --input-filetype "jsonl" \
    --output-path /path/to/clusters \
    --text-field "text" \
    --id-field "climb_id"
```

At least 1 GPU is required to run this step. Use the `--num-cpus` and `--num-gpus` arguments as desired to control the number of CPUs and GPUs used by the Ray client. The script uses all available resources by default.

The script uses `--n-clusters 1000` as the default. See script for full list of K-Means parameters.

## Step 3: Cluster Pruning

Use a FastText model to prune the created clusters:

```bash
python 3_cluster_pruning.py
```

## Step 4: Generate Training Data Mixtures

Generate a mixture of data ratios to be used for training a proxy model:

```bash
python 4_synthesize_mixture.py
```

## Step 5: Train Proxy Model

Kick off a Megatron training job with the specified data ratios:

```bash
# TODO: Add script
```

## Step 6: Evaluate Proxy Model

Evaluate the proxy model using NeMo Evaluator:

```bash
# TODO: Add script
```

## Step 7: Train Predictor

Train a LightGBM model on the results with:

```bash
python 7_predictor_training.py
```
