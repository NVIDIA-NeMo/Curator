# TTS Granary annotation

Annotate JSONL manifests that already have inverse-text-normalized text
(`tn_raw` or `itn_text`) with IPA (espeak-ng) and optional TTS Granary
audio stages (Sortformer speaker ID, UTMOS, bandwidth, SED, early cut-off).

```bash
# IPA only (CPU). Requires espeak-ng on PATH.
python tutorials/audio/granary_hifi/run.py \
  --input_manifest /path/to/manifest.jsonl \
  --output_manifest /tmp/tts_out.jsonl

python tutorials/audio/granary_hifi/run.py \
  --input_manifest /path/to/manifest.jsonl \
  --output_manifest /tmp/tts_out.jsonl \
  --enable_bandwidth --enable_mos --enable_speaker_id

# Early cut-off endpoint gate. bundle_dir holds the v13 checkpoint files.
python tutorials/audio/granary_hifi/run.py \
  --input_manifest /path/to/manifest.jsonl \
  --output_manifest /tmp/tts_out.jsonl \
  --enable_early_cut_off \
  --early_cut_off_bundle_dir /path/to/early_cut_off_detection/v13
```
