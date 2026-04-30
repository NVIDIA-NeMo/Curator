# Speaker Workflow

This tutorial runs a tarred speaker-embedding workflow on top of diarized
manifests that already contain `diarized_segments`.

The workflow performs:

1. `TarredAudioManifestReader`
2. `MaterializeTarredAudioStage`
3. `SpeakerEmbeddingAudioTaskStage`
4. `BuildUploadKeyStage`
5. `UploadFilesStage`
6. `CleanupTemporaryAudioStage`

It is driven by Hydra YAML and uses `nemo_curator.config.run.create_pipeline_from_yaml`.

## Layout

This workflow lives under:

- `tutorials/audio/hifi_pipeline/speaker_workflow/0_embeddings/`

## Usage

From the Curator repo root:

```bash
python tutorials/audio/hifi_pipeline/speaker_workflow/0_embeddings/main.py \
  --config-path tutorials/audio/hifi_pipeline/speaker_workflow/0_embeddings \
  --config-name embeddings_workflow \
  manifest_paths=s3://bucket/path/to/manifest__OP_0..255_CL_.json \
  tar_paths=s3://bucket/path/to/audio__OP_0..255_CL_.tar \
  output_dir=/tmp/speaker_workflow \
  upload_bucket=my-output-bucket
```

## Optional checkpointing

To run with audio checkpointing:

```bash
python tutorials/audio/hifi_pipeline/speaker_workflow/0_embeddings/main.py \
  --config-path tutorials/audio/hifi_pipeline/speaker_workflow/0_embeddings \
  --config-name embeddings_workflow \
  manifest_paths=s3://bucket/path/to/manifest__OP_0..255_CL_.json \
  tar_paths=s3://bucket/path/to/audio__OP_0..255_CL_.tar \
  output_dir=/tmp/speaker_workflow \
  upload_bucket=my-output-bucket \
  checkpoint_dir=/tmp/speaker_workflow_checkpoints \
  link_stages_via_io=true
```

## Notes

- `output_dir` stores local NPZ artifacts under `output_dir/npz`
- uploaded S3 object keys are built from the manifest `output_key` field plus
  the configured `upload_key_prefix`
- `CleanupTemporaryAudioStage` cleans temporary local audio files, but uploaded
  NPZ artifacts remain
