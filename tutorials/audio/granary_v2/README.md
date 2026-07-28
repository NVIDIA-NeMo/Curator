# Granary v2 production ASR pipeline

This tutorial wires the independent Granary-v2 feature stages into one
production contract:

`reader → initialization → optional SED → primary ASR → hallucination check → optional recovery ASR → recovery check → prediction selection → regex normalization → abbreviation concatenation → sharded writer`.

The model route is derived from `language` by
`nemo_curator.stages.audio.pipeline_utils.resolve_model_route`. Explicit
`primary_model` and `recovery_model` values override the defaults.
Every model route uses the shared `ASRStage`; Qwen-Omni, Qwen3-ASR,
NeMo/FastConformer, local `.nemo`, IndicConformer, and Faster-Whisper differ
only by `adapter_target`. The reader's in-memory waveform is normalized once
by that shared stage and retained only when a recovery model still needs it.

There is intentionally no second ASR turn. The configuration never supplies a
follow-up prompt, never creates a second-turn prediction column, and does not
include `DisfluencyWerGuardStage`.

## Run

```bash
python tutorials/audio/granary_v2/main.py \
  input_config=/data/input.yaml \
  output_dir=/data/granary_v2 \
  language=en \
  models.parakeet_riva=/models/parakeet-riva.nemo
```

Pass `sed_checkpoint=/models/Cnn14.pth` to insert the SED inference and
postprocessing pair. Leave it null to omit both stages.

The input YAML follows the NeMo Speech `input_cfg` contract. Output shards
mirror their stable input-relative keys and gain a sibling `.jsonl.done`
marker only after the expected number of rows has been written. Re-running the
same command skips completed shards and recovers partial writer counts.

The configuration is source-mergeable independently because stage targets are
stored as Hydra strings. It becomes runnable after the corresponding stage PRs
land: production I/O, initialization, routing, model adapters, hallucination
filtering, prediction selection, regex substitution, abbreviation
concatenation, and the optional SED pair.
