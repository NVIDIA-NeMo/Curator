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

"""Hydra entry point for the Granary v2 Qwen-Omni in-process pipeline.

This wraps the same pipeline logic as
``examples/audio/qwen_omni_inprocess/run_pipeline.py`` behind a Hydra
config so it can be invoked by NvLLMOps Kratos workflows via::

    python tutorials/audio/qwen_omni_inprocess/main.py \\
        --config-path=<path> --config-name=qwen_omni_inprocess \\
        workspace_dir=/work input_manifest=/data/config.yaml

The NvLLMOps ``run_curator()`` function calls::

    tutorials/audio/{pipeline_to_run}/main.py --config-path=... --config-name=...

with Hydra overrides ``workspace_dir``, ``input_manifest``,
``language_short``, ``max_segment_length``, ``hf_token``, and
``final_manifest``.  All of these are accepted as top-level keys in the
YAML and forwarded to the appropriate stages.
"""

import os
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.inference.qwen_asr import InferenceQwenASRStage
from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.stages.audio.io.nemo_tarred_reader import NemoTarredAudioReader
from nemo_curator.stages.audio.io.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.stages.audio.text_filtering import (
    AbbreviationConcatStage,
    DisfluencyWerGuardStage,
    FastTextLIDStage,
    InitializeFieldsStage,
    ITNRestorationStage,
    PnCContentGuardStage,
    PnCRestorationStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
)
from nemo_curator.stages.audio.text_filtering.select_best_prediction import SelectBestPredictionStage
from nemo_curator.stages.resources import Resources


def _read_file_or_str(value: str | None) -> str | None:
    """Return file contents if *value* is a readable path, else return it as-is."""
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            return f.read().strip()
    return value


def build_granary_v2_pipeline(cfg: DictConfig) -> Pipeline:  # noqa: C901, PLR0912, PLR0915
    """Construct the full Granary v2 stage chain from a Hydra config."""
    data_config = cfg.get("data_config") or cfg.get("input_manifest")
    if not data_config:
        raise ValueError("Either 'data_config' or 'input_manifest' must be set in the config")

    output_dir = cfg.get("output_dir") or cfg.get("workspace_dir", "./output")

    corpus_filter = OmegaConf.to_container(cfg.corpus, resolve=True) if "corpus" in cfg and cfg.corpus else None

    ml_prompt = _read_file_or_str(cfg.get("ml_prompt_file")) or cfg.get("ml_prompt", "Transcribe the audio.")
    en_prompt = _read_file_or_str(cfg.get("en_prompt_file"))
    followup_prompt = _read_file_or_str(cfg.get("followup_prompt_file")) or cfg.get("followup_prompt")
    system_prompt = _read_file_or_str(cfg.get("system_prompt"))
    pnc_prompt = _read_file_or_str(cfg.get("pnc_prompt_file")) or cfg.get("pnc_prompt")
    itn_prompt = _read_file_or_str(cfg.get("itn_prompt_file"))

    omni_text_key = "qwen3_prediction_s2" if followup_prompt else "qwen3_prediction_s1"
    source_lang_key = cfg.get("source_lang_key", "source_lang")
    asr_model_id = cfg.get("asr_model_id")

    stages = [
        NemoTarredAudioReader(
            yaml_path=data_config,
            corpus_filter=corpus_filter,
            s3_endpoint_url=cfg.get("s3_endpoint_url"),
            output_dir=output_dir,
        ).with_({"nemo_tar_shard_reader": {"resources": Resources(cpus=4.0)}}),

        InitializeFieldsStage(),

        InferenceQwenOmniStage(
            model_id=cfg.get("model_id", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
            prompt_text=ml_prompt,
            en_prompt_text=en_prompt,
            followup_prompt=followup_prompt,
            system_prompt=system_prompt,
            tensor_parallel_size=cfg.get("tensor_parallel_size", 2),
            batch_size=cfg.get("batch_size", 32),
            max_output_tokens=cfg.get("max_output_tokens", 256),
            max_model_len=cfg.get("max_model_len", 32768),
            max_num_seqs=cfg.get("max_num_seqs", 16),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.95),
            prep_workers=cfg.get("prep_workers", 16),
            source_lang_key=source_lang_key,
            pred_text_key="qwen3_prediction_s1",
            disfluency_text_key="qwen3_prediction_s2",
            keep_waveform=bool(asr_model_id),
        ),
    ]

    if followup_prompt:
        stages.append(DisfluencyWerGuardStage(
            ref_text_key="qwen3_prediction_s1",
            hyp_text_key="qwen3_prediction_s2",
            max_wer_pct=cfg.get("max_wer_pct", 50.0),
        ))

    stages.append(WhisperHallucinationStage(
        name="WhisperHallucination_omni",
        common_hall_file=cfg.hall_phrases,
        text_key=omni_text_key,
        unique_words_threshold=cfg.get("unique_words_threshold", 0.4),
        long_word_threshold=cfg.get("long_word_threshold", 25),
        long_word_rel_threshold=cfg.get("long_word_rel_threshold", 3.0),
        max_char_rate=cfg.get("max_char_rate", 40.0),
    ))

    if asr_model_id:
        stages.extend([
            InferenceQwenASRStage(
                model_id=asr_model_id,
                source_lang_key=source_lang_key,
                batch_size=cfg.get("asr_batch_size", 128),
                gpu_memory_utilization=cfg.get("asr_gpu_memory_utilization", 0.95),
                max_new_tokens=cfg.get("asr_max_new_tokens", 4096),
            ),
            WhisperHallucinationStage(
                name="WhisperHallucination_asr",
                common_hall_file=cfg.hall_phrases,
                text_key="qwen3_asr_prediction",
                overwrite=True,
                recovery_value="Recovered:QwenASR",
                unique_words_threshold=cfg.get("unique_words_threshold", 0.4),
                long_word_threshold=cfg.get("long_word_threshold", 25),
                long_word_rel_threshold=cfg.get("long_word_rel_threshold", 3.0),
                max_char_rate=cfg.get("max_char_rate", 40.0),
            ),
        ])

    stages.append(SelectBestPredictionStage(
        primary_text_key=omni_text_key,
        asr_text_key="qwen3_asr_prediction",
    ))

    stages.extend([
        FastTextLIDStage(
            model_path=cfg.get("fasttext_model", "facebook/fasttext-language-identification"),
            text_key="best_prediction",
            source_lang_key=source_lang_key,
            min_lang_prob=cfg.get("min_lang_prob", 0.8),
        ),
        RegexSubstitutionStage(
            regex_params_yaml=cfg.regex_yaml,
            text_key="best_prediction",
            output_text_key="cleaned_text",
        ),
        AbbreviationConcatStage(
            text_key="cleaned_text",
            output_text_key="abbreviated_text",
            source_lang_key=source_lang_key,
        ),
    ])

    if not cfg.get("skip_pnc", False):
        pnc_kwargs = {}
        if pnc_prompt:
            pnc_kwargs["pnc_prompt"] = pnc_prompt
        if cfg.get("completeness_prompt"):
            pnc_kwargs["completeness_prompt"] = cfg.completeness_prompt
        stages.extend([
            PnCRestorationStage(
                model_id=cfg.get("pnc_model_id", "Qwen/Qwen3.5-35B-A3B-FP8"),
                text_key="abbreviated_text",
                output_text_key="pnc_text",
                tensor_parallel_size=cfg.get("pnc_tensor_parallel_size", 2),
                batch_size=cfg.get("pnc_batch_size", 64),
                max_model_len=cfg.get("pnc_max_model_len", 8192),
                max_num_seqs=cfg.get("pnc_max_num_seqs", 64),
                gpu_memory_utilization=cfg.get("pnc_gpu_memory_utilization", 0.95),
                prep_workers=cfg.get("pnc_prep_workers", 8),
                **pnc_kwargs,
            ),
            PnCContentGuardStage(
                text_key="abbreviated_text",
                pnc_text_key="pnc_text",
                rejected_text_key="rejected_pnc_text",
            ),
        ])

    if cfg.get("enable_itn", False):
        skip_pnc = cfg.get("skip_pnc", False)
        itn_text_key = cfg.get("itn_text_key") or ("pnc_text" if not skip_pnc else "abbreviated_text")
        stages.append(ITNRestorationStage(
            model_id=cfg.get("itn_model_id", "Qwen/Qwen3.5-35B-A3B-FP8"),
            prompt_text=itn_prompt,
            text_key=itn_text_key,
            output_text_key=cfg.get("itn_output_key", "itn_text"),
            tensor_parallel_size=cfg.get("itn_tensor_parallel_size"),
            max_output_tokens=cfg.get("itn_max_output_tokens", 4096),
            max_model_len=cfg.get("itn_max_model_len", 4096),
            max_num_seqs=cfg.get("itn_max_num_seqs", 16),
            gpu_memory_utilization=cfg.get("itn_gpu_memory_utilization", 0.95),
            batch_size=cfg.get("itn_batch_size", 64),
            enable_validation=not cfg.get("itn_no_validation", False),
        ))

    stages.append(ShardedManifestWriterStage(output_dir=output_dir))

    return Pipeline(name="qwen_omni_inference", stages=stages)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the Granary v2 Qwen-Omni pipeline."""
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    hf_token = cfg.get("hf_token", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ.setdefault("HF_HOME", "/tmp/hf_home")

    pipeline = build_granary_v2_pipeline(cfg)
    logger.info(f"Pipeline: {pipeline.describe()}")

    executor = XennaExecutor(
        config={"execution_mode": cfg.get("execution_mode", "streaming")}
    )

    t0 = time.time()
    pipeline.run(executor=executor)
    elapsed = time.time() - t0
    output_dir = cfg.get("output_dir") or cfg.get("workspace_dir", "./output")
    logger.info(f"Pipeline finished in {elapsed / 60:.1f} min. Output: {output_dir}")


if __name__ == "__main__":
    main()
