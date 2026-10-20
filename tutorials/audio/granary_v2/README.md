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
