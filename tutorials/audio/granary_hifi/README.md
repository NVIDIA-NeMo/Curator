# GranaryHiFi: Granary ASR manifests → contextual TTS training shards

Curator implementation of the GranaryHiFi pipeline ("GranaryHiFi: A 203K-Hour Open Dataset for Contextual Speech
Synthesis in 24 European Languages"). It turns Granary sample manifests (one JSONL row per source recording with a
`segments` list, segment audio addressed as tar members via `GranaryHifi.AudioMetadata`) into Lhotse SHAR shards with
target/context audio, IPA, 25 fps codec codes and the training flags consumed by EasyMagpie / Magpie-TTS.

| Paper step | Stage | Writes (per segment) |
|---|---|---|
| 4.2 diarization | `SegmentDiarizationStage` (streaming Sortformer) | `diarization.{num_speakers, diarized_speech_rate, ...}` |
| 4.2 speaker clustering | `SpeakerClusteringStage` (TitaNet, agglomerative cosine 0.4) | `speaker_cluster.{speaker_id, cluster_similarity, cluster_margin}` |
| 4.2 sound events | `SegmentSEDStage` (CNN14 framewise + GBM postprocessor) | `sed.{weighted_speech_rate, weighted_model_confidence, weighted_classifier_confidence}` |
| 4.2 quality | `UTMOSv2AnnotationStage`, `BandwidthAnnotationStage` | `utmosv2.score`, `bandwidth.bandwidth` |
| 4.2 enhancement (optional) | `StudioVoiceEnhancementStage` (Maxine StudioVoice, Triton) or `ResembleEnhanceStage` (ReUse) | `enhanced_audio.{audio_filepath, sampling_rate, method}` |
| 4.3 segment filter | `SegmentFilterStage(preset=ytc\|yodas\|voxpopuli\|librilight)` | drops failing segments |
| 4.4 context selection | `ContextSelectionStage` (same speaker, ≥3 s, sim ≥0.6, overlap ≤0.5) | `context.*` |
| 4.6 phonemization | `ManifestIpaStage` (espeak-ng, from `tn_raw`) | `ipa.ipa` |
| 4.5 end-of-word check | `EarlyCutOffEndpointGateStage` → `EarlyCutOffMFAResolverStage` | `early_cut_off_detection.decision` (OK/DROP/REVIEW) |
| 4.6 training flags | `TrainingFlagsStage` | `tts_flags.{valid_tn_raw_transcript, valid_ipa, valid_context_audio_text, early_cut_off, context_audio_text_overlap}` |
| 4.6 artifacts | `LhotseSharBuildStage` → `CodecCodesStage` → `SharValidationStage` → `publish_shars` | SHAR shards, codes, validation report, published layout |

Annotation stages never drop rows; failures are recorded in each annotation's `error`. Training-time filtering
(`valid_* == true`, `early_cut_off == false`, `context_audio_text_overlap < 0.5`) is left to the data loader, as in the
released dataset; `nemo_curator.stages.audio.tts.shar.apply_training_filter` mirrors it.

## Requirements

- `nemo_curator[audio_cuda12]` (NeMo, lhotse, torchlibrosa), plus `utmosv2` (UTMOSv2) and `tritonclient[grpc]`
  (StudioVoice only).
- `espeak-ng` 1.52 on `PATH` (IPA), Montreal Forced Aligner 3.4.1 with the per-language models listed in
  `early_cutoff/mfa_resolver.py` (MFA step), and the v13 endpoint bundle (`english_endpoint.pt`,
  `multilingual_endpoint.pt`, calibrations, `manifest.json`).
- Model files: Sortformer (`nvidia/diar_streaming_sortformer_4spk-v2.1` from HF by default; the published dataset used
  `diar_streaming_sortformer_4spk-v2.nemo`), TitaNet large, `Cnn14_DecisionLevelMax_mAP=0.385.pth`,
  `gbt_multiclass_model.joblib`, UTMOSv2 weights, `25fps_spectral_codec_with_bandwidth_extension.nemo`.
- ReUse enhancement: a separate venv with `resemble-enhance` (`--resemble_python`) and its `enhancer_stage2` weights
  (`--resemble_run_dir`; otherwise resemble downloads them with git-lfs). Use `numpy<2` in that venv.
- Running on a single workstation: Xenna detects GPUs with `nvidia-ml-py` (`pynvml`) and only counts the ones listed
  in `CUDA_VISIBLE_DEVICES`, so install it and set e.g. `CUDA_VISIBLE_DEVICES=0,1`. With fewer GPUs than GPU stages,
  pass `--execution_mode batch` (stages run one after another instead of all at once).

## Run

```bash
# 1. annotate + filter + context + IPA + early cut-off + flags  (Sec. 4.2-4.6)
python tutorials/audio/granary_hifi/run.py annotate --preset ytc \
  --input_manifest /data/ytc/it/shard_0.jsonl --output_manifest /out/ytc_it_shard_0.annotated.jsonl \
  --tar_root /data/ytc/audios_with_orig_sr --strip_tar_key_prefix tts_granary/audios_with_orig_sr/ytc \
  --sortformer_path diar_streaming_sortformer_4spk-v2.nemo --titanet_path speakerverification_en_titanet_large.nemo \
  --sed_checkpoint Cnn14_DecisionLevelMax_mAP=0.385.pth --sed_classifier gbt_multiclass_model.joblib \
  --utmos_checkpoint fold0_s42_best_model.pth \
  --early_cut_off_bundle_dir early_cut_off_v13/ --mfa_root_dir /opt/mfa

# 2. SHAR + codec codes + validation + hard-link publish  (Sec. 4.6, 4.1)
python tutorials/audio/granary_hifi/run.py build_shar \
  --input_manifest /out/ytc_it_shard_0.annotated.jsonl --shar_dir /out/shar \
  --dataset "GranaryHiFi_YTC (orig_audio)" --language it \
  --codec_model 25fps_spectral_codec_with_bandwidth_extension.nemo --publish_root /out/published
```

### Enhanced vs original audio

One switch: `annotate --enhancement studio_voice|reuse` enhances every segment first, and **all following steps**
(diarization, speaker embeddings/clustering, SED, UTMOSv2, bandwidth, context selection, end-of-word check) and the
shards built by `build_shar` (target/context audio and codec codes) then use the enhanced audio. With
`--enhancement none` (default, as in the paper) everything uses the original audio. `build_shar` detects the choice from
the annotated manifest. Segments whose enhancement failed are skipped unless `--enhanced_fallback` (use original audio).
Shards keep every speaker in one shard (with its context clips); `SharValidationStage` checks this.

Steps can be skipped with `--skip` (e.g. `--skip studio_voice mfa_resolver`). StudioVoice runs only when
`--studio_voice_url` points at a running Maxine StudioVoice Triton server (see `studio_voice.py`).
