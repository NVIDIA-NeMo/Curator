# GranaryHiFi: Granary ASR manifests → TTS training shards

Turns Granary manifests (one JSONL row per recording, `segments` with tar-member audio in `GranaryHifi.AudioMetadata`)
into Lhotse SHAR shards with target/context audio, IPA, neural codec codes and training flags.

| Step | Stage | Writes (per segment) |
|---|---|---|
| enhancement (optional) | `StudioVoiceEnhancementStage` or `ResembleEnhanceStage` | `enhanced_audio.*` |
| diarization | `SegmentDiarizationStage` | `diarization.{num_speakers, diarized_speech_rate, ...}` |
| speaker clustering | `SpeakerClusteringStage` | `speaker_cluster.{speaker_id, cluster_similarity, cluster_margin}` |
| sound events | `SegmentSEDStage` | `sed.weighted_*` |
| quality | `UTMOSv2AnnotationStage`, `BandwidthEstimationStage` | `utmosv2.score`, `bandwidth.bandwidth` |
| segment filter | `SegmentFilterStage(preset=ytc\|yodas\|voxpopuli\|librilight)` | drops failing segments |
| context selection | `ContextSelectionStage` | `context.*` |
| phonemization | `ManifestIpaStage` | `ipa.ipa` |
| end-of-word check | `EarlyCutOffEndpointGateStage` → `EarlyCutOffMFAResolverStage` | `early_cut_off_detection.decision` |
| training flags | `TrainingFlagsStage` | `tts_flags.*` |
| shards | `LhotseSharBuildStage` → `CodecCodesStage` → `SharValidationStage` → `publish_shars` | SHAR shards, codec codes |

Only the segment filter drops segments. Training-time filtering (`valid_* == true`, `early_cut_off == false`,
`context_audio_text_overlap < 0.5`) is left to the data loader; `shar.apply_training_filter` mirrors it.

## Requirements

- `uv sync --extra audio_tts`, plus UTMOSv2: `uv pip install git+https://github.com/sarulab-speech/UTMOSv2.git`.
- `espeak-ng` (1.51 or 1.52) and Montreal Forced Aligner 3.4.1 on `PATH`; the early cut-off endpoint bundle.
- Model files: `gbt_multiclass_model.joblib` (SED classifier); the other models download automatically.
- Codec: either the public `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` model from Hugging Face or a local
  `25fps_spectral_codec_with_bandwidth_extension.nemo`.
- ReUse: a separate `resemble-enhance` venv (`--resemble_python`, `numpy<2`) and its weights (`--resemble_run_dir`).
- Single workstation: set `CUDA_VISIBLE_DEVICES` and use `--execution_mode batch` if there are fewer GPUs than GPU
  stages.

## Run

```bash
python tutorials/audio/granary_hifi/run.py annotate --preset ytc \
  --input_manifest in.jsonl --output_manifest annotated/ --tar_root /data/audio \
  --sed_classifier gbt_multiclass_model.joblib \
  --early_cut_off_bundle_dir endpoint_bundle/ --mfa_root_dir /opt/mfa

# Public 21.5 fps NeMo NanoCodec (downloaded from Hugging Face).
python tutorials/audio/granary_hifi/run.py build_shar --preset ytc --language it \
  --input_manifest 'annotated/*.jsonl' --tar_root /data/audio --shar_dir shar/ \
  --codec_model nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps --publish_root published/

# Or use a local 25 fps spectral codec.
python tutorials/audio/granary_hifi/run.py build_shar --preset ytc --language it \
  --input_manifest 'annotated/*.jsonl' --tar_root /data/audio --shar_dir shar/ \
  --codec_model 25fps_spectral_codec_with_bandwidth_extension.nemo --publish_root published/
```

`--enhancement studio_voice|reuse` enhances every segment first; all later steps and the shards then use the enhanced
audio (`build_shar` detects it). Default `none` uses the original audio.

`--preset default` uses the general filter thresholds (the same as `yodas`). All flags: `run.py annotate --help`,
`run.py build_shar --help`.

`annotate --skip` takes step names: `enhancement diarization speaker_clustering sed utmos bandwidth filter context ipa
early_cut_off mfa_resolver training_flags`. The `librilight` preset uses the row-level `speaker_id` instead of
speaker clustering. `build_shar --skip_codec` skips codec codes. `--codec_model` accepts either a Hugging Face model
ID or a local `.nemo` path; codec frame rate is read from the loaded model.

`annotate --output_manifest` is an output directory. The final `AudioToDocumentStage` removes cached waveform tensors
and `JsonlWriter` writes parallel JSONL shards into that directory.

## Input

One JSONL row per recording (shown indented; dotted names elsewhere in this README are nested keys):

```json
{"orig_sample_id": "rec1", "orig_sample_rate": 16000, "source_lang": "it",
 "segments": [{"duration": 8.2,
               "GranaryV2": {"pnc_text": "Penso che sia importante.", "tn_raw": "Penso che sia importante."},
               "GranaryHifi": {"AudioMetadata": {"tar_key": "audio/rec1.tar", "tar_member": "rec1_0.wav",
                                                 "tar_member_offset": 512, "tar_member_bytes": 526050}}}]}
```

`tar_member_offset` is the byte offset of the member data (`TarInfo.offset_data`); relative `tar_key`s are resolved
with `--tar_root`.
