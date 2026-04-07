# Nemotron-Climb Data Curation

TODO: Description here

## Step 0: Requirements

The following Python libraries are needed to run this tutorial:

- `nemo-curator`
- `lightgbm`
- `seaborn`
- `xformers`

TODO: Add hardware requirements

## Step 1: Compute Embeddings

Compute text embeddings on the dataset with:

```bash
python 1_embed.py \
    --input-path /path/to/input/data/dir \
    --input-filetype "jsonl" \
    --output-path /path/to/computed_embeddings \
    --text-field "text" \
    --id-field "climb_id" \
    --use-sentence-transformer
```

At least 1 GPU is required to run this step. Use the `--num-cpus` and `--num-gpus` arguments as desired to control the number of CPUs and GPUs used by the Ray client. The script uses all available resources by default.

If the dataset already contains a unique ID field, feel free to omit the `--id-field` argument.

The script uses the [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) embedding model by default. The [SentenceTransformers](https://huggingface.co/sentence-transformers) library is used via the `use-sentence-transformer` flag to enhance performance.

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
    --centroids-path /path/to/centroids
```

At least 1 GPU is required to run this step. Use the `--num-cpus` and `--num-gpus` arguments as desired to control the number of CPUs and GPUs used by the Ray client. The script uses all available resources by default.

The script uses `--n-clusters 1000` as the default. The `--id-field` generated from step 1 (or an existing ID column) is required. See script for full list of K-Means parameters.

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
    --output-path /path/to/filtered_clusters \
    --fasttext-model-paths ${FASTTEXT_MODEL_PATHS[@]} \
    --score-fields ${FASTTEXT_SCORE_FIELDS[@]} \
    --text-field "text" \
    --centroids-path /path/to/centroids
```

No GPUs are needed to run this step, so by default the script sets `--num-gpus 0`. Use the `--num-cpus` argument as desired to control the number of CPUs used by the Ray client; by default, all are used.

There are 5 FastText quality models that can be used for this step. Each can be pre-downloaded from Hugging Face:

TODO: Update links when the models are published

- [best_model_advertisement.bin](https://huggingface.co/nvidia)
- [best_model_cultural_value.bin](https://huggingface.co/nvidia)
- [best_model_educational_value.bin](https://huggingface.co/nvidia)
- [best_model_informational_value.bin](https://huggingface.co/nvidia)
- [best_model_quality.bin](https://huggingface.co/nvidia)

Users may opt to run the script with all 5 models as demonstrated above, or a subset of the models. For each path in `--fasttext-model-paths`, a unique score field must be set via the `--score-fields` argument.

After the FastText scores are computed, clusters with an average score less than `--pruning-threshold 1.0` are removed. Finally, remaining clusters with a Euclidean distance closer than `--merge-threshold 1.5` are combined with each other.

Because each FastText model is large, CPU out-of-memory errors may occur due to overhead between stage workers. Try decreasing the number of CPUs if needed.

## Step 4: Generate Training Data Mixtures

Generate a mixture of data ratios to be used for training a proxy model:

```bash
python 4_mixture.py
```

TODO: Add more information

## Step 5: Train Proxy Model

Kick off a Megatron training job with the specified data ratios:

```bash
# TODO: Add script
```

TODO: Add more information

## Step 6: Evaluate Proxy Model

Evaluate the proxy model using NeMo Evaluator:

```bash
# TODO: Add script
```

TODO: Add more information

## Step 7: Train Predictor

Train a LightGBM model on the results with:

```bash
python 7_predict.py
```

TODO: Add more information
