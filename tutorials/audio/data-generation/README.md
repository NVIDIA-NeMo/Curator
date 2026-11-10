# Multi-speaker Conversation Generation

Generate conversation audio from topics with the shared YAML runner:

1. `ManifestReader` loads topic rows.
2. `vLLMInference` generates and validates structured conversation turns.
3. `ChatterboxTTSStage` synthesizes the turns with consistent speaker voices.
4. `MFAAlignmentStage` aligns their transcripts.
5. A `group_by: conversation_id` boundary gathers each complete conversation.
6. `MergeConversationSDPStage` mixes audio and annotations; `ManifestWriterStage` writes the final JSONL.

The generation and merge phases both use the standard executor interfaces. Scheduler batch sizes never define conversation boundaries. No tutorial-specific runner is needed.

Each phase must contain a stage pipeline or nested pipeline phases. Workflows remain available as standalone YAML configurations; workflow result containers cannot cross phase boundaries. Invalid workflow phases are rejected before any phase runs.

## Requirements

```bash
uv sync --extra audio_cuda12 --extra vllm
source .venv/bin/activate
```

Chatterbox uses its separate Ray runtime environment so its Torch/Transformers requirements do not conflict with Curator's vLLM environment. Reference voices and output paths must be accessible to every worker.

Install MFA separately with conda or micromamba, then download models to the same root used for alignment:

```bash
export MFA_ROOT_DIR=/data/mfa
/path/to/mfa/env/bin/mfa model download acoustic english_us_arpa
/path/to/mfa/env/bin/mfa model download dictionary english_us_arpa
# Optional, for out-of-vocabulary words:
/path/to/mfa/env/bin/mfa model download g2p english_us_arpa
```

Use a CUDA GPU configuration that fits the selected models. The default generation backend, `ray_actor_pool`, releases each stage's actors before starting the next stage, allowing LLM and TTS to use the same GPU in sequence. Set `llm_tensor_parallel_size` to reserve multiple GPUs for the LLM. Ray Data and Xenna are also supported; simultaneous GPU stages require enough devices for all resident models.

## Run

From the repository root:

```bash
python nemo_curator/config/run.py \
    --config-path ../../tutorials/audio/data-generation \
    --config-name pipeline \
    input_manifest="$PWD/tutorials/audio/data-generation/topics/topics_dialog.jsonl" \
    output_dir=/data/conversations/run1 \
    reference_voices_dataset=/data/reference_voices \
    prompt_file="$PWD/tutorials/audio/data-generation/prompts/dialog_prompt.yaml" \
    mfa_command=/path/to/mfa/env/bin/mfa \
    mfa_root_dir=/data/mfa
```

Choose a fresh output directory for each run: existing conversation destinations are refused rather than reused.

| Prompt | Speakers |
|---|---|
| `prompts/dialog_prompt.yaml` | `speaker_0`, `speaker_1` |
| `prompts/triperson_prompt.yaml` | `speaker_0` through `speaker_2` |
| `prompts/fourperson_prompt.yaml` | `speaker_0` through `speaker_3` |

Each prompt receives the JSON Schema used for constrained generation and validation. Extra fields, wrong types, unknown speakers, non-finite or out-of-range overlaps, and conversations that omit a configured speaker are rejected and retried. Default overlap bounds are -2.0 to 2.5 seconds. Positive overlap requests an interruption; negative overlap inserts a pause.

Input rows contain a topic:

```json
{"topic": "Planning a weekend trip"}
```

Conversation identity includes canonical turn content and timing plus input row identity and data. Duplicate topic rows remain separate conversations. Each generated turn carries the expected conversation length so incomplete groups cannot be published.

## Outputs

The final manifest is `merged_manifest.jsonl`. Each conversation has `mixed.wav`, `multichannel.wav`, `all.rttm`, optional `all.ctm`, and `segments.seglst.json`. Per-speaker files use generated names under `speakers/`. The result's `speaker_artifacts` mapping preserves original labels and provides file paths and zero-based channel indices. Missing CTM outputs have empty paths.

Task metadata retains member metadata, task IDs, and available source IDs across grouping. Grouped merging is non-resumable because it combines source lineages. Unreadable audio, inconsistent sample rates, incomplete generated groups, or failed writes reject the affected conversation. Complete artifact directories are published together; other valid groups can continue.

## Configuration and validation

`pipeline.yaml` exposes LLM sampling and tensor parallelism, TTS sample rate and synthesis settings, MFA command/root/job count/timeout, and merge pause limits. Select executors with `phases.0.backend` and `phases.1.backend`; configure Xenna execution mode within its phase.

Tests cover runner lifecycle and failure propagation, prompt variants, validation/retry, conversation identity, provenance, and the full flow through each executor with lightweight model/synthesis substitutes, real WAVs, TextGrid conversion, merging, and manifest writing. Running the neural models themselves requires compatible CUDA hardware and downloaded weights.
