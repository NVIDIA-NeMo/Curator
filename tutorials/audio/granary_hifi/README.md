# GranaryHiFi: Granary ASR manifests → TTS training shards

Turns Granary manifests (one JSONL row per recording, `segments` with tar-member audio in `GranaryHifi.AudioMetadata`)
into Lhotse SHAR shards with target/context audio, IPA, 25 fps codec codes and training flags.

| Step | Stage | Writes (per segment) |
|---|---|---|
| enhancement (optional) | `StudioVoiceEnhancementStage` or `ResembleEnhanceStage` | `enhanced_audio.*` |
| diarization | `SegmentDiarizationStage` | `diarization.{num_speakers, diarized_speech_rate, ...}` |
| speaker clustering | `SpeakerClusteringStage` | `speaker_cluster.{speaker_id, cluster_similarity, cluster_margin}` |
| sound events | `SegmentSEDStage` | `sed.weighted_*` |
| quality | `UTMOSv2AnnotationStage`, `BandwidthAnnotationStage` | `utmosv2.score`, `bandwidth.bandwidth` |
| segment filter | `SegmentFilterStage(preset=ytc\|yodas\|voxpopuli\|librilight)` | drops failing segments |
| context selection | `ContextSelectionStage` | `context.*` |
| phonemization | `ManifestIpaStage` | `ipa.ipa` |
| end-of-word check | `EarlyCutOffEndpointGateStage` → `EarlyCutOffMFAResolverStage` | `early_cut_off_detection.decision` |
| training flags | `TrainingFlagsStage` | `tts_flags.*` |
| shards | `LhotseSharBuildStage` → `CodecCodesStage` → `SharValidationStage` → `publish_shars` | SHAR shards, codec codes |

Only the segment filter drops segments. Training-time filtering (`valid_* == true`, `early_cut_off == false`,
`context_audio_text_overlap < 0.5`) is left to the data loader; `shar.apply_training_filter` mirrors it.

## Requirements

- `nemo_curator[audio_cuda12]`; `utmosv2`; `tritonclient[grpc]` for StudioVoice.
- `espeak-ng` 1.52 and Montreal Forced Aligner 3.4.1 on `PATH`; the early cut-off endpoint bundle.
- Model files: TitaNet large, `Cnn14_DecisionLevelMax_mAP=0.385.pth`, `gbt_multiclass_model.joblib`, UTMOSv2 weights,
  `25fps_spectral_codec_with_bandwidth_extension.nemo`.
- ReUse: a separate `resemble-enhance` venv (`--resemble_python`, `numpy<2`) and its weights (`--resemble_run_dir`).
- Single workstation: install `nvidia-ml-py`, set `CUDA_VISIBLE_DEVICES`, and use `--execution_mode batch` if there are
  fewer GPUs than GPU stages.

## Run

```bash
python tutorials/audio/granary_hifi/run.py annotate --preset ytc \
  --input_manifest in.jsonl --output_manifest annotated.jsonl --tar_root /data/audio \
  --titanet_path speakerverification_en_titanet_large.nemo \
  --sed_checkpoint Cnn14_DecisionLevelMax_mAP=0.385.pth --sed_classifier gbt_multiclass_model.joblib \
  --early_cut_off_bundle_dir endpoint_bundle/ --mfa_root_dir /opt/mfa

python tutorials/audio/granary_hifi/run.py build_shar --preset ytc --language it \
  --input_manifest annotated.jsonl --shar_dir shar/ \
  --codec_model 25fps_spectral_codec_with_bandwidth_extension.nemo --publish_root published/
```

`--enhancement studio_voice|reuse` enhances every segment first; all later steps and the shards then use the enhanced
audio (`build_shar` detects it). Default `none` uses the original audio.

`annotate --skip` takes step names: `enhancement diarization speaker_clustering sed utmos bandwidth filter context ipa
early_cut_off mfa_resolver training_flags`. The `librilight` preset uses the row-level `speaker_id` instead of
speaker clustering. `build_shar --skip_codec` skips codec codes.

## Input

One JSONL row per recording:

- `orig_sample_id`, `orig_sample_rate`, `source_lang`;
- `segments`: list of `{duration, GranaryV2.pnc_text, GranaryV2.tn_raw, GranaryHifi.AudioMetadata.{tar_key,
  tar_member, tar_member_offset, tar_member_bytes}}` (relative `tar_key`s are resolved with `--tar_root`).
