# Nemotron-Climb Data Curation

TODO: Description here

## Step 0: Requirements

- `nemo-curator`
- `lightgbm`
- `seaborn`

## Step 1: Compute Embeddings
```bash
python 1_compute_embeddings.py \
    --input-path /path/to/input/data/dir \
    --input-file-type "jsonl" \
    --output-path /path/to/computed_embeddings \
    --output-file-type "jsonl" \
    --id-field "climb_id"
```

## Step 2: K-Means Clustering
```bash
python 2_clustering.py \
    --input-path /path/to/computed_embeddings \
    --input-filetype "jsonl" \
    --output-path /path/to/clusters \
    --id-field "climb_id" \
    --metadata-fields "text"
```

## Step 3: Cluster Pruning
```bash
python 3_cluster_pruning.py
```

## Step 4: Generate Training Data Mixtures
```bash
python 4_synthesize_mixture.py
```

## Step 5: Train Proxy Model
```bash
# TODO: Add script
```

## Step 6: Evaluate Proxy Model
```bash
# TODO: Add script
```

## Step 7: Train Predictor
```bash
python 7_predictor_training.py
```
