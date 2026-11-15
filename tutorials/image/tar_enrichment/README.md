# Annotate images stored inside TARs

Use this when a Parquet catalog records where each image lives inside WebDataset-style TARs
(on S3 or any fsspec filesystem): one row per image with the TAR path, byte offset and byte size.
The pipeline adds CLIP embeddings, aesthetic and NSFW scores, and CPU measurements
(dimensions, sharpness, QR-code coverage) without dropping any image.

```text
catalog Parquet
  → ImageTarRangeReader   one ranged GET per group of up to 128 images in the same TAR
  → ImageDecodeStage      CPU: decode, EXIF orientation, alpha over white
  → ImageMeasurementStage CPU: dimensions, sharpness, QR coverage on native pixels
  → ImageResizeStage      CPU: 224x224 crop, so GPUs only receive small uint8 arrays
  → ImageEmbeddingStage → aesthetic and NSFW scores (score_threshold=None: annotate, don't filter)
  → Parquet
```

## Run

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...   # or an AWS profile
python tutorials/image/tar_enrichment/main.py \
  --input /path/to/catalog/ --output /path/to/annotated/ \
  --model-dir /path/to/models --endpoint-url https://my-s3-endpoint \
  --tar-column storage_uri --offset-column byte_offset --size-column byte_size --id-column image_url
```

`--model-dir` must contain `openai/clip-vit-large-patch14`, `ttj/sac-logos-ava1-l14-linearMSE` and
`laion/clip-autokeras-binary-nsfw`; missing weights are downloaded on first use.

## Output

One row per catalog row: `image_id`, `metadata` (the catalog columns plus
`image_decode_status`, `image_width`, `image_height`, sharpness and QR fields with their statuses),
`embedding` (768 floats), `aesthetic_score` and `nsfw_score`.
Images that could not be fetched or decoded keep their row with null scores;
`image_decode_status` says why (`missing`, `decode_error` or `resource_limit`).

## Performance

On one node (4 GPUs, 108 CPUs) with the default pool sizes, 248k web images from 30 TARs
(28 GiB on an S3-compatible store) took 255 s end to end, about 2,000 images/s in steady state.

- `--fetch-workers`: object stores often cap per-connection throughput, so run many concurrent
  ranged reads. Each fetch reserves a quarter CPU.
- `--cpu-workers`: decode, measurement and resize are fused into one CPU task, so native-resolution
  pixels never cross the object store.
- `--clip-workers`: CLIP actors use half a GPU each and TF32 matmuls (about 5.6x faster than strict
  FP32, with the same embeddings).
- The reader sorts rows by TAR and offset itself. Selecting whole TARs keeps each ranged read dense;
  a sparse selection also downloads the bytes between selected images.
