# Sound Event Detection (SED)

Label each utterance in an audio manifest with the sound events it contains
(speech, music, and several noise categories) using an AudioSet-pretrained
[CNN14 / PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) model.

## Pipeline

| # | Stage | Task flow | Output |
|---|-------|-----------|--------|
| 1 | `ManifestReader` | `∅ → AudioTask` | one task per manifest line |
| 2 | `_LoadWaveformStage` *(local)* | `AudioTask → AudioTask` | in-memory `waveform` + `sample_rate` |
| 3 | `SEDInferenceStage` | `AudioTask → AudioTask` | `_sed_framewise` `(T, 527)`, `sed_fps`, `sed_valid_frames` |
| 4 | `SEDPostprocessingStage` | `AudioTask → AudioTask` | `sed_events` (labeled spans) |
| 5 | `_KeepEventsStage` *(local)* | `AudioTask → AudioTask` | drops transient arrays |
| 6 | `AudioToDocumentStage` + `JsonlWriter` | `AudioTask → DocumentBatch → ∅` | labeled JSONL manifest |

Stages 3 and 4 are the reusable SED stages added in this branch
(`nemo_curator.stages.audio.inference.SEDInferenceStage` and
`nemo_curator.stages.audio.postprocessing.SEDPostprocessingStage`). Stages 2 and
5 are small helpers defined inline in `pipeline.py` to load audio and to strip
the large framewise/waveform arrays before serialization.

## Prerequisites

- A JSONL manifest where each line has an `audio_filepath`.
- A PANNs `Cnn14_DecisionLevelMax` checkpoint (`.pth`). The stage does **not**
  download weights — pass the path explicitly. The public checkpoint
  (`Cnn14_DecisionLevelMax_mAP=0.385.pth`) is available from the
  [PANNs release](https://zenodo.org/record/3987831).
- The audio curation extras installed: `pip install -e ".[audio_cuda12]"`
  (or `.[audio_cpu]`), which now includes `torchlibrosa`.

## Run

```bash
python pipeline.py \
    --input-manifest /data/manifest.jsonl \
    --checkpoint     /models/Cnn14_DecisionLevelMax_mAP=0.385.pth \
    --output-dir     /data/sed_out \
    --threshold      0.5 \
    --min-duration-sec 0.3
```

Add `--emit-subcategories` to label each event with its individual AudioSet
class (and parent superclass) instead of the superclass group.

## Output

Each line of the output manifest carries the original fields plus `sed_events`:

```json
{
  "audio_filepath": "/data/utt_001.wav",
  "sed_events": [
    {"start_time": 2.0, "end_time": 4.0, "mean_confidence": 0.91, "max_confidence": 0.98, "label": "speech"},
    {"start_time": 5.1, "end_time": 6.4, "mean_confidence": 0.62, "max_confidence": 0.77, "label": "music"}
  ]
}
```

See the [Sound Event Detection docs](../../../fern/versions/v26.04/pages/curate-audio/process-data/quality-filtering/sound-event-detection.mdx)
for parameter details and threshold-tuning guidance.
