# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import hydra
import numpy as np
import pytest

from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.tasks import AudioTask

PIPELINE_DIR = Path(__file__).resolve().parents[3] / "tutorials/audio/granary_v2/full_pipeline"
CONFIGS = [
    "en_granary_pipeline",
    "european_qwen_omni_granary_pipeline",
    "european_parakeet_v3_granary_pipeline",
    "european_whisper_granary_pipeline",
    "qwen_asr_whisper_granary_pipeline",
    "hebrew_whisper_granary_pipeline",
    "indic_granary_pipeline",
    "indic_english_granary_pipeline",
]


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """Configuration tests do not start a Ray cluster."""


@pytest.fixture(autouse=True)
def h100_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a GPU the memory lookup falls back to 24 GB, too small for the Indic Transcribe slices."""
    monkeypatch.setattr("nemo_curator.stages.resources._get_gpu_memory_gb", lambda: 80.0)


@pytest.fixture(autouse=True)
def checkout_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompts search path, regex rules and hallucination phrases are paths from the checkout root."""
    monkeypatch.chdir(PIPELINE_DIR.parents[3])


def _compose(config_name: str, *overrides: str):  # noqa: ANN202
    with hydra.initialize_config_dir(version_base=None, config_dir=str(PIPELINE_DIR / "pipelines")):
        return hydra.compose(
            config_name=config_name,
            overrides=[
                "manifest_path=/data/in.jsonl",
                "model_dir=/models",
                "output_dir=/out",
                "cache_root=/cache",
                "endpoint=http://inference-host:8000/v1",
                *overrides,
            ],
        )


@pytest.mark.parametrize("config_name", CONFIGS)
def test_every_pipeline_yaml_builds(config_name: str) -> None:
    stages = create_pipeline_from_yaml(_compose(config_name), log_config=False).stages

    assert stages
    assert len({stage.name for stage in stages}) == len(stages)


MAX_MODEL_LEN = {"en_granary_pipeline": 2048, "indic_english_granary_pipeline": 2048, "indic_granary_pipeline": 11264}


@pytest.mark.parametrize(("config_name", "max_model_len"), [(name, MAX_MODEL_LEN.get(name, 3072)) for name in CONFIGS])
def test_text_model_context_and_workers(config_name: str, max_model_len: int) -> None:
    cfg = _compose(config_name)
    engine = cfg.inference_server.models[0].engine_kwargs
    scaling = cfg.inference_server.models[0].deployment_config.autoscaling_config
    stages = create_pipeline_from_yaml(cfg, log_config=False).stages
    text_stages = [stage for stage in stages if type(stage).__name__ == "AudioTextLLMStage"]

    assert (scaling.min_replicas, scaling.max_replicas) == (1 if config_name.startswith("indic") else 0, 4)
    assert engine.max_model_len == max_model_len
    assert engine.tensor_parallel_size == 2  # two GPUs per replica, four replicas: one 8-GPU node
    # Chunked prefill, async scheduling and eager mode stay at vLLM's defaults.
    assert not {"enable_chunked_prefill", "async_scheduling", "enforce_eager"} & set(engine)
    assert len(text_stages) == 6
    assert all(stage.num_workers() == 12 for stage in text_stages)


@pytest.mark.parametrize("config_name", CONFIGS)
def test_hallucination_phrases_match_strictly(config_name: str) -> None:
    stages = create_pipeline_from_yaml(_compose(config_name), log_config=False).stages
    checks = [stage for stage in stages if type(stage).__name__ == "WhisperHallucinationStage"]

    assert checks
    assert not any(stage.loose_phrase_match for stage in checks)


@pytest.mark.parametrize("config_name", CONFIGS)
def test_manifest_writers_refuse_an_earlier_run(config_name: str) -> None:
    cfg = _compose(config_name)
    stages = create_pipeline_from_yaml(cfg, log_config=False).stages
    writers = [stage for stage in stages if type(stage).__name__ == "ManifestWriterStage"]

    assert [writer.output_path for writer in writers] == ["/out/metadata.jsonl", cfg.output_path]
    assert all(writer.mode == "error" for writer in writers)


@pytest.mark.parametrize("config_name", CONFIGS)
def test_rejected_speech_qa_request_writes_skip(config_name: str) -> None:
    stages = create_pipeline_from_yaml(_compose(config_name), log_config=False).stages
    fused = next(stage for stage in stages if stage.name == "FusedTextLLM")
    speech_qa = next(op for op in fused.operation_stages() if op.output_text_key == "speech_qa_text")

    assert speech_qa.generation_error_text == "SKIP"


@pytest.mark.parametrize(("config_name", "isolate"), [(name, name == "indic_granary_pipeline") for name in CONFIGS])
def test_only_indic_text_isolates_failed_requests(config_name: str, isolate: bool) -> None:
    # A failed English/European text request fails the batch; the Indic pipeline isolates the failed row.
    stages = create_pipeline_from_yaml(_compose(config_name), log_config=False).stages
    text_stages = [stage for stage in stages if type(stage).__name__ == "AudioTextLLMStage"]

    assert text_stages
    assert all(
        op.isolate_failed_requests is isolate for stage in text_stages for op in [stage, *stage.operation_stages()]
    )


@pytest.mark.parametrize(
    ("config_name", "language", "prompt_start"),
    [
        ("en_granary_pipeline", "en", "You receive audio in English.\n\nMAIN GOAL: faithfully transcribe"),
        ("indic_english_granary_pipeline", "en", "You receive audio in English.\n\nMAIN GOAL: faithfully transcribe"),
        (
            "european_qwen_omni_granary_pipeline",
            "de",
            "Transcribe the German audio into text exactly as the speaker says it. Write numbers as spoken words.",
        ),
    ],
)
def test_omni_asr_prompt_follows_the_row_language(config_name: str, language: str, prompt_start: str) -> None:
    stages = create_pipeline_from_yaml(_compose(config_name), log_config=False).stages
    omni = next(stage for stage in stages if getattr(stage, "adapter_target", "").endswith(".QwenOmniASRAdapter"))
    language_name = omni._resolve_language(AudioTask(data={omni.source_lang_key: language}))

    assert omni._create_adapter()._get_prompt_text(language_name).startswith(prompt_start)


@pytest.mark.parametrize("config_name", CONFIGS)
def test_code_switching_prompt_suffix_rule_per_pipeline(config_name: str) -> None:
    stages = create_pipeline_from_yaml(_compose(config_name), log_config=False).stages
    fused = next(stage for stage in stages if stage.name == "FusedTextLLM")
    prompt = next(op for op in fused.operation_stages() if op.output_text_key == "code_switched_text").prompt
    indic = config_name == "indic_granary_pipeline"

    # English and European attach native suffixes to restored English roots; Indic separates them with one space.
    assert ("Do NOT insert a space, hyphen, or any separator" in prompt) is not indic
    assert ("separated by exactly one space" in prompt) is indic


_SPEECH = "so we went to the market and bought some apples"


@pytest.mark.parametrize(
    ("config_name", "language"),
    [
        ("en_granary_pipeline", "en"),
        ("european_qwen_omni_granary_pipeline", "de"),
    ],
)
@pytest.mark.parametrize(("primary", "recovered", "source"), [(_SPEECH, "", "primary"), ("", _SPEECH, "fallback")])
def test_selection_keeps_the_non_empty_transcript(
    config_name: str, language: str, primary: str, recovered: str, source: str
) -> None:
    stages = {stage.name: stage for stage in create_pipeline_from_yaml(_compose(config_name), log_config=False).stages}
    recovery = stages["QwenASR_recovery"]
    adapter = recovery._create_adapter()
    adapter._model = MagicMock()
    adapter._model.transcribe.return_value = [SimpleNamespace(text=recovered, language="")]
    recovery._adapter = adapter
    data = {
        "source_lang": language,
        "duration": 3.0,
        "sample_rate": 16000,
        "waveform": np.zeros(48000, dtype=np.float32),
        "primary_model_prediction": primary,
        "_skipme": "",
    }

    tasks = recovery.process_batch([AudioTask(data=data)])
    for name in ("WhisperHallucination_asr", "SelectBestPrediction", "RegexSubstitution", "AbbreviationConcat"):
        stages[name].setup()
        tasks = stages[name].process_batch(tasks)

    assert tasks[0].data["fallback_model_prediction"] == recovered
    assert tasks[0].data["_skipme"] == ""
    assert tasks[0].data["best_prediction_source"] == source
    assert tasks[0].data["abbreviated_text"] == _SPEECH


def _asr_stages(config_name: str) -> dict:
    stages = create_pipeline_from_yaml(_compose(config_name), log_config=False).stages
    return {stage.name: stage for stage in stages if type(stage).__name__ == "ASRStage"}


def test_filipino_runs_qwen_asr_then_whisper() -> None:
    stages = _asr_stages("qwen_asr_whisper_granary_pipeline")

    assert stages["QwenASR_primary"].supported_language_codes == ["th", "fa", "tl"]
    assert stages["Whisper_recovery"].supported_language_codes == ["th", "fa", "tl"]
    for config_name in CONFIGS:
        gate = _compose(config_name).stages
        kept = {c["target_value"] for s in gate if "conditions" in s for c in s["conditions"]}
        assert not kept & {"fil", "ms"}, config_name


def test_hebrew_runs_whisper_without_recovery() -> None:
    stages = _asr_stages("hebrew_whisper_granary_pipeline")

    assert list(stages) == ["Whisper_primary"]
    assert stages["Whisper_primary"].model_id == "large-v3"
    assert stages["Whisper_primary"].supported_language_codes == ["he"]
    assert stages["Whisper_primary"].keep_waveform is False


@pytest.mark.parametrize("config_name", CONFIGS)
def test_regex_rule_set_per_pipeline(config_name: str) -> None:
    stages = create_pipeline_from_yaml(_compose(config_name), log_config=False).stages
    regex = next(stage for stage in stages if stage.name == "RegexSubstitution")

    # Indic recordings are cleaned with the Indic rules, English included.
    assert regex.rule_set == ("indic" if config_name.startswith("indic") else "default")
    regex.setup()


@pytest.mark.parametrize("config_name", CONFIGS)
def test_flagged_pair_agreement_threshold(config_name: str) -> None:
    stages = create_pipeline_from_yaml(_compose(config_name), log_config=False).stages
    select = next(stage for stage in stages if stage.name == "SelectBestPrediction")

    # A doubly flagged primary is kept at 80% agreement, SelectBestPredictionStage's default.
    assert select.min_agreement_pct == 80.0


@pytest.mark.parametrize("config_name", CONFIGS)
def test_no_asr_stage_splits_a_vad_clip(config_name: str) -> None:
    # VAD caps clips at max_duration_sec; an ASRStage splits only audio longer than max_inference_duration_s.
    stages = create_pipeline_from_yaml(_compose(config_name), log_config=False).stages
    vad = next(stage for stage in stages if type(stage).__name__ == "VADSegmentationStage")
    asr_stages = [stage for stage in stages if isinstance(stage, ASRStage)]

    assert asr_stages
    assert all(stage.max_inference_duration_s >= vad.max_duration_sec for stage in asr_stages)


def test_english_in_indic_recordings_runs_omni_then_indic_transcribe() -> None:
    cfg = _compose("indic_english_granary_pipeline")
    stages = create_pipeline_from_yaml(cfg, log_config=False).stages
    by_name = {stage.name: stage for stage in stages}
    gate = next(stage for stage in cfg.stages if "conditions" in stage)
    omni, recovery = by_name["QwenOmni_primary"], by_name["IndicTranscribe_recovery"]
    text_env = cfg.inference_server.models[0].runtime_env

    assert [condition["target_value"] for condition in gate["conditions"]] == ["en"]
    assert omni.resources.gpus == 1
    assert omni.supported_language_codes == ["en"]
    # The Indic image has only vLLM's CPU build: Omni and the text model run vLLM 0.22 in isolated environments.
    assert {"vllm==0.22.0+cu129", "torchaudio==2.11.0"} <= set(omni.runtime_env["uv"]["packages"])
    assert omni.runtime_env["env_vars"]["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert "vllm==0.22.0+cu129" in text_env.uv.packages
    assert recovery.pred_text_key == "fallback_model_prediction"
    assert cfg.model_identifier == "Qwen/Qwen3.6-27B-FP8"
    assert by_name["ITNRestoration"].validation_mode == "itn"


INDIC_LANGUAGES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur",
                   "brx", "doi", "kok", "ks", "mai", "mni", "ne", "sa", "sat", "sd"]  # fmt: skip


def test_indic_itn_has_rules_for_every_indic_language() -> None:
    cfg = _compose("indic_granary_pipeline")
    itn = next(s for s in create_pipeline_from_yaml(cfg, log_config=False).stages if s.name == "ITNRestoration")
    rules = itn.language_rules

    assert itn.validation_mode == "none"
    assert sorted(rules) == sorted(INDIC_LANGUAGES)
    assert all(str(rule).strip() for rule in rules.values())
    assert cfg.prompts.itn_restoration_indic.count("{language_rules}") == 1


LID_MODELS = [
    ("nemo_curator.models.audio.text_lid.FastTextLIDAdapter", "/models/lid.176.bin"),
    ("nemo_curator.models.audio.text_lid.IndicLIDAdapter", "/models/indiclid-ftn/model_baseline_roman.bin"),
]
FASTTEXT_LANGUAGES = ["bn", "hi", "kn", "mr", "pa", "ta", "te"]
INDICLID_LANGUAGES = ["as", "brx", "doi", "gu", "kok", "ks", "mai", "ml", "mni", "ne", "or", "sa", "sat", "ur"]


def _language_id_stages(config_name: str, *overrides: str) -> list:
    stages = create_pipeline_from_yaml(_compose(config_name, *overrides), log_config=False).stages
    start = next(index for index, stage in enumerate(stages) if stage.name == "PnCRestoration") + 1
    end = next(index for index, stage in enumerate(stages) if stage.name == "TextNormalization")
    return stages[start:end]


def _llm_languages(llm: object) -> list[str]:
    return [language for language in INDIC_LANGUAGES if llm._is_source_language_enabled({llm.language_key: language})]


@pytest.mark.parametrize("config_name", CONFIGS)
def test_language_id_stages_run_after_pnc_and_the_llm_checks_every_row(config_name: str) -> None:
    language_id = _language_id_stages(config_name)
    cpu = language_id[1:]

    expected = ["LanguageID"] + ["FastTextLanguageID", "IndicLIDLanguageID"] * (
        config_name == "indic_granary_pipeline"
    )
    assert [stage.name for stage in language_id] == expected
    assert type(language_id[0]).__name__ == "AudioTextLLMStage"
    assert all(type(stage).__name__ == "TextLanguageIDStage" for stage in cpu)
    assert all(stage.output_text_key == "llm_language_prediction" for stage in language_id)
    assert _llm_languages(language_id[0]) == INDIC_LANGUAGES
    assert [(stage.adapter_target, stage.adapter_kwargs["model_path"]) for stage in cpu] == LID_MODELS[: len(cpu)]
    assert all(stage.text_key == "pnc_text" and stage.languages == [] for stage in cpu)


def test_language_id_split_sends_listed_languages_to_the_cpu_models() -> None:
    llm, fasttext, indiclid = _language_id_stages(
        "indic_granary_pipeline",
        f"language_id_split.fasttext=[{','.join(FASTTEXT_LANGUAGES)}]",
        f"language_id_split.indiclid=[{','.join(INDICLID_LANGUAGES)}]",
    )

    assert fasttext.languages == FASTTEXT_LANGUAGES
    assert indiclid.languages == INDICLID_LANGUAGES
    assert _llm_languages(llm) == ["sd"]


def test_indic_pnc_prompt_is_an_option_with_rules_for_every_indic_language() -> None:
    cfg = _compose("indic_granary_pipeline")
    generic = next(s for s in create_pipeline_from_yaml(cfg, log_config=False).stages if s.name == "PnCRestoration")
    indic_cfg = _compose("indic_granary_pipeline", "pnc_prompt=indic")
    indic = next(
        s for s in create_pipeline_from_yaml(indic_cfg, log_config=False).stages if s.name == "PnCRestoration"
    )

    # One rule per Indic language, in a fixed order, and one {language_rules} placeholder in the prompt.
    assert list(cfg.prompts.pnc_language_rules) == INDIC_LANGUAGES
    assert all(str(rule).strip() for rule in cfg.prompts.pnc_language_rules.values())
    assert cfg.prompts.pnc_restoration_indic.count("{language_rules}") == 1
    assert (generic.prompt, generic.language_rules) == (cfg.prompts.pnc_restoration, None)
    assert indic.prompt == cfg.prompts.pnc_restoration_indic
    assert sorted(indic.language_rules) == sorted(INDIC_LANGUAGES)
    assert (indic.text_key, indic.output_text_key, indic.validation_mode) == (
        generic.text_key,
        generic.output_text_key,
        generic.validation_mode,
    )


@pytest.mark.parametrize(("overrides", "expected"), [((), False), (("ignore_skip_me=true",), True)])
def test_ignore_skip_me_reaches_every_text_stage(overrides: tuple[str, ...], expected: bool) -> None:
    stages = create_pipeline_from_yaml(_compose("indic_granary_pipeline", *overrides), log_config=False).stages
    text_stages = [stage for stage in stages if type(stage).__name__ == "AudioTextLLMStage"]

    assert all(op.ignore_skip_me is expected for stage in text_stages for op in [stage, *stage.operation_stages()])
