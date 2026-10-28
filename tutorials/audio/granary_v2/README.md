# Granary v2 audio tutorials

Prepare speech clips and metadata, choose an ASR pipeline, then enrich the
transcripts. Each tutorial contains its complete pipeline YAML and documents its
environment, sample data, input and output fields, and validation commands.

| Step | Tutorial | Result |
| --- | --- | --- |
| Prepare recordings | [Metadata extraction](metadata/README.md) | Opus speech clips with speaker, sound-event and language metadata |
| Transcribe Hindi clips | [Indic ASR](indic_asr/README.md) | Canary and Parakeet predictions, selected and cleaned transcripts, and WER |
| Transcribe with Qwen3-Omni | [Qwen Omni ASR](qwen_omni_asr/README.md) | Transcripts checked for hallucinations and cleaned for downstream use |
| Process transcripts | [Text pipeline](text_pipeline/README.md) | Entity recovery, punctuation, language verification, normalization and text features |

Choose the ASR tutorial that matches your language and models. Follow each
tutorial's input contract when passing manifests between steps; the bundled
examples can also be run independently.

Run commands from the repository root. All four tutorials use
`nemo_curator/config/run.py`; their READMEs provide the configuration paths and
required arguments. Environment and hardware requirements are specific to each
tutorial.

The [benchmarks](../../../benchmarking/README.md#audio-benchmark-data-setup)
remain under `benchmarking/` and use these tutorial configurations on larger
datasets.

## Observed throughput

The latest completed EOS comparison ran on 2026-10-20 using Curator
`8d275057`. Each Xenna/Ray Data pair used the same frozen input cohort,
tutorial environment, models and stage settings, with eight NVIDIA H100
80 GB GPUs allocated per run.

| Pipeline | Benchmark cohort | Xenna streaming | Ray Data |
| --- | --- | ---: | ---: |
| Metadata | 30,631 recordings / 107.76357 source audio hours | 472.309 audio-hours/hour | 166.251 audio-hours/hour |
| Indic ASR | 216,164 Hindi clips / 531.75763 audio hours | 3,604.471 audio-hours/hour | 387.649 audio-hours/hour |
| Qwen Omni ASR | 22,710 English clips / 80.61259 audio hours | 406.615 audio-hours/hour | 228.424 audio-hours/hour |
| Text pipeline | 14,290 English transcript rows | 30.283 rows/second | 12.217 rows/second |

These whole-node rates use pipeline time, not full-process time. The text
pipeline reused an already-running server; its 510.097-second startup is
excluded. Audio pipelines include model startup. Each tutorial's throughput
section gives the clocks, per-allocated-GPU rates and output checks.
All inputs are accounted for, including retained flagged rows, but generated
outputs differ between executors. These measurements compare each executor
pair; different models, cohorts and timing boundaries prevent treating the
four pipelines as a common speed or accuracy ranking.
