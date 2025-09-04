# Getting Started with Image Curation

The Python scripts in this directory contain examples for how to run typical image curation workflows with NeMo Curator. In particular:

- `image_curation_example.py` implements a pipeline to read images, generate their [CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip) embeddings, score and filter the images by aesthetic and NSFW content, and save the results
- `image_dedup_example.py` implements a pipeline to read images, generate their [CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip) embeddings, filter semantic duplicates (i.e., images which look similar in content to other images within the dataset), and save the results
- `helper.py` contains functions for downloading and saving image data used by the above scripts

Note: Run these examples on GPUs for best performance.

### Download and preprocess data

Use the following commands to download a sample Parquet of image URLs from MSCOCO dataset and prepare a deduplicated (removing duplicating URLs), truncated subset for quick experiments.

```bash
mkdir -p ./example_data/ && wget https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet -O ./example_data/mscoco.parquet
```

```python
import pandas as pd

NUM_URLS = 100_000
urls = pd.read_parquet("./example_data/mscoco.parquet")
deduplicated_urls = urls[~urls["URL"].duplicated()]
truncated_urls = deduplicated_urls[:NUM_URLS]
truncated_urls.to_parquet("./example_data/truncated_100k_mscoco.parquet")
```

# Download and prepare relevant models weights


```bash
# LAION NSFW detector weights
mkdir -p ./model_weights/laion/clip-autokeras-binary-nsfw && \
wget -qO ./model_weights/laion/clip-autokeras-binary-nsfw/clip_autokeras_binary_nsfw.zip \
  https://github.com/LAION-AI/CLIP-based-NSFW-Detector/files/10250461/clip_autokeras_binary_nsfw.zip && \
unzip -o ./model_weights/laion/clip-autokeras-binary-nsfw/clip_autokeras_binary_nsfw.zip -d ./model_weights/laion/clip-autokeras-binary-nsfw

# OpenAI CLIP ViT-L/14 weights
mkdir -p ./model_weights/openai && \
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-large-patch14', local_dir='./model_weights/openai/clip-vit-large-patch14', force_download=True)"

# Aesthetic predictor weights
mkdir -p ./model_weights/ttj && \
python -c "from huggingface_hub import snapshot_download; snapshot_download('ttj/sac-logos-ava1-l14-linearMSE', local_dir='./model_weights/ttj/sac-logos-ava1-l14-linearMSE', force_download=True)"
```

### Run the scripts

Run the image curation pipeline on GPUs (extracting embeddings, NSFW and aesthetics scores, filtering based on thresholds):

```bash
python tutorials/image/getting-started/image_curation_example.py \
    --input-parquet ./example_data/truncated_100k_mscoco.parquet \
    --input-wds-dataset-dir ./example_data/truncated_100k_mscoco \
    --output-dataset-dir ./example_data/results_truncated_100k_mscoco \
    --model-dir ./model_weights \
    --tar-files-per-partition 10 \
    --task-batch-size 500 \
    --embedding-batch-size 500 \
    --aesthetic-batch-size 500 \
    --nsfw-batch-size 500 \
    --aesthetic-threshold 0.9 \
    --nsfw-threshold 0.9 \
    --images-per-tar 1000 \
    --verbose
```

Run the image deduplication pipeline on GPUs (extracting embeddings, running semantic deduplication, removing duplicated samples):

```bash
python tutorials/image/getting-started/image_dedup_example.py \
    --input-wds-dataset-dir ./example_data/truncated_100k_mscoco \
    --output-dataset-dir ./example_data/results_truncated_100k_mscoco \
    --embeddings-dir ./example_data/dedup/embeddings/truncated_100k_mscoco \
    --removal-parquets-dir ./example_data/dedup/removal_ids/truncated_100k_mscoco \
    --model-dir ./model_weights \
    --task-batch-size 1000 \
    --embedding-batch-size 500 \
    --tar-files-per-partition 10 \
    --skip-download \
    --verbose
```
