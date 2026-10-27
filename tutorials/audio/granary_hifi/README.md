# GranaryHiFi: Granary ASR manifests → annotated, filtered manifests

Turns Granary manifests (one JSONL row per recording, `segments` with tar-member audio in `GranaryHifi.AudioMetadata`)
into annotated JSONL manifests: speaker, quality and sound-event annotations, a same-speaker context clip, IPA,
end-of-word verdicts and training flags, with the segments that fail any gate dropped.

Packing the result into Lhotse SHAR shards and extracting neural codec codes are available as Curator stages
(`LhotseSharBuildStage`, `CodecCodesStage`, `SharValidationStage`) but are not part of this pipeline: no audio is
read out of the source tars and rewritten.

| Step | Stage | Writes (per segment) |
|---|---|---|
| diarization | `SegmentDiarizationStage` | `diarization.{num_speakers, diarized_speech_rate, ...}` |
| speaker clustering | `SpeakerClusteringStage` | `speaker_cluster.{speaker_id, cluster_similarity, cluster_margin}` |
| sound events | `SegmentSEDStage` | `sed.weighted_*` |
| quality | `UTMOSv2AnnotationStage`, `BandwidthEstimationStage` | `utmosv2.score`, `bandwidth.bandwidth` |
| segment filter | `SegmentFilterStage(preset=ytc\|yodas\|voxpopuli\|librilight)` | drops failing segments |
| context selection | `ContextSelectionStage` | `context.*` |
| phonemization | `ManifestIpaStage` | `ipa.ipa` |
| end-of-word check | `EarlyCutOffEndpointGateStage` → `EarlyCutOffMFAResolverStage` | `early_cut_off_detection.decision` |
| training flags | `TrainingFlagsStage` | `tts_flags.*` |
| training filter | `SegmentFilterStage(conditions=training_conditions(), context_key="context")` | drops failing segments |
| enhancement (optional) | `StudioVoiceEnhancementStage` or `ResembleEnhanceStage` | `enhanced_audio.*` |

Two steps drop segments: the segment filter and the training filter (`valid_* == true`, `early_cut_off == false`,
`context_audio_text_overlap < 0.5`, the data loader's predicate). The training filter keeps a failing segment that a kept
segment uses as context, so its audio stays available; its flags still exclude it as a training target.

## Requirements

- `uv sync --extra audio_tts`, plus UTMOSv2: `uv pip install git+https://github.com/sarulab-speech/UTMOSv2.git`.
- `espeak-ng` (1.51 or 1.52) and Montreal Forced Aligner 3.4.1 on `PATH`.
- Every model downloads on first use and is cached under `~/.cache/nemo_curator/granary_hifi/` or the Hugging Face
  cache: Sortformer, TitaNet, UTMOSv2, CNN14, the wav2vec2 CTC backbone and MFA models, plus the SED classifier and
  the early cut-off endpoint bundle from
  [`nvidia/Granary-HiFi`](https://huggingface.co/datasets/nvidia/Granary-HiFi/tree/main/assets/quality_control).
  `--sed_classifier` and `--early_cut_off_bundle_dir` use local copies instead.
- ReUse: a separate `resemble-enhance` venv (`--resemble_python`, `numpy<2`); its weights are downloaded from
  Hugging Face unless `--resemble_run_dir` points at a pre-staged copy.
- Single workstation: set `CUDA_VISIBLE_DEVICES` and use `--execution_mode batch` if there are fewer GPUs than GPU
  stages.

## Run

```bash
python tutorials/audio/granary_hifi/run.py annotate --preset ytc \
  --input_manifest in.jsonl --output_manifest annotated/ --tar_root /data/audio --mfa_root_dir /opt/mfa
```

`--enhancement studio_voice|reuse` runs last: every annotation and filter uses the original audio, then the segments
that pass all filters (and their context clips) are enhanced and recorded under `enhanced_audio`. Default `none`
uses the original audio.

`--preset default` uses the general filter thresholds (the same as `yodas`). All flags: `run.py annotate --help`.

`annotate --skip` takes step names: `diarization speaker_clustering sed utmos bandwidth filter context ipa early_cut_off
mfa_resolver training_flags training_filter enhancement`. The `librilight` preset uses the row-level `speaker_id`
instead of speaker clustering.

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
