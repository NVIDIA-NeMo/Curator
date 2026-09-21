# TTS Granary annotation

Annotate JSONL manifests that already have inverse-text-normalized text
(`tn_raw` or `itn_text`) with IPA (espeak-ng) and optional TTS Granary
audio stages (Sortformer speaker ID, UTMOS, bandwidth, SED).

```bash
# IPA only (CPU). Requires espeak-ng on PATH.
python tutorials/audio/tts_granary/run.py \
  --input_manifest /path/to/manifest.jsonl \
  --output_manifest /tmp/tts_out.jsonl

python tutorials/audio/tts_granary/run.py \
  --input_manifest /path/to/manifest.jsonl \
  --output_manifest /tmp/tts_out.jsonl \
  --enable_bandwidth --enable_mos --enable_speaker_id
```
