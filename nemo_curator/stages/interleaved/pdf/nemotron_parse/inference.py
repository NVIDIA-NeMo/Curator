# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""GPU inference stage for Nemotron-Parse."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import pyarrow as pa
import torch
from loguru import logger
from PIL import Image

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch

DEFAULT_MODEL_PATH = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
PROMPT_BASE = "</s><s><predict_bbox><predict_classes><output_markdown>"


def build_task_prompt(*, text_in_pic: bool = False) -> str:
    """Build the Nemotron-Parse task prompt with the appropriate text-in-pic token."""
    suffix = "<predict_text_in_pic>" if text_in_pic else "<predict_no_text_in_pic>"
    return f"{PROMPT_BASE}{suffix}"


@dataclass
class NemotronParseInferenceStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """GPU stage: run Nemotron-Parse inference on pre-rendered page images.

    Reads PNG page images from ``binary_content``, runs model inference, and
    writes raw Nemotron-Parse output into ``text_content``.

    Supports two inference backends:

    - ``"vllm"`` (recommended): vLLM offline mode with continuous batching.
      Batching is handled internally by vLLM via ``max_num_seqs``.
    - ``"hf"``: HuggingFace Transformers with manual micro-batching via
      ``inference_batch_size``.

    Parameters
    ----------
    model_path
        HuggingFace model ID or local path (e.g. ``nvidia/NVIDIA-Nemotron-Parse-v1.2``).
    text_in_pic
        Whether to predict text inside pictures. When ``True``, uses the
        ``<predict_text_in_pic>`` prompt token; when ``False`` (default), uses
        ``<predict_no_text_in_pic>``. Only applies to Nemotron-Parse v1.2+.
    task_prompt
        Override the full prompt string. When set, ``text_in_pic`` is ignored.
    backend
        Inference backend: ``"vllm"`` or ``"hf"``.
    inference_batch_size
        Pages per GPU forward pass (HF backend only).
    max_num_seqs
        Maximum concurrent sequences (vLLM backend only).
    """

    model_path: str = DEFAULT_MODEL_PATH
    text_in_pic: bool = False
    task_prompt: str | None = None
    backend: str = "vllm"
    inference_batch_size: int = 4
    max_num_seqs: int = 64
    enforce_eager: bool = False
    name: str = "nemotron_parse_inference"
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0, gpus=1.0))

    def __post_init__(self) -> None:
        if self.task_prompt is None:
            self.task_prompt = build_task_prompt(text_in_pic=self.text_in_pic)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    # -- setup / teardown --

    def setup_on_node(self, node_info: dict | None = None, worker_metadata: dict | None = None) -> None:  # noqa: ARG002
        """Initialize model once per node (serially) to avoid torch.compile race conditions."""
        self._initialize_model()

    def setup(self, worker_metadata: dict | None = None) -> None:  # noqa: ARG002
        if not (hasattr(self, "_llm") or hasattr(self, "_model")):
            self._initialize_model()

    def _initialize_model(self) -> None:
        if self.backend == "vllm":
            self._setup_vllm()
        else:
            self._setup_hf()

    def _setup_hf(self) -> None:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig

        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        logger.info(f"[HF] Loading {self.model_path} on {device}")
        self._device = device
        self._model = (
            AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._gen_config = GenerationConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self._proc_size: tuple[int, int] = tuple(self._processor.image_processor.final_size)
        logger.info(f"[HF] Model loaded, proc_size={self._proc_size}")

    def _setup_vllm(self) -> None:
        from vllm import SamplingParams

        from nemo_curator.utils.vllm_utils import create_vllm_llm, resolve_local_model_path

        resolved_path = resolve_local_model_path(self.model_path)
        self._llm = create_vllm_llm(
            resolved_path,
            max_num_seqs=self.max_num_seqs,
            enforce_eager=self.enforce_eager,
        )
        self._sampling_params = SamplingParams(
            temperature=0,
            top_k=1,
            repetition_penalty=1.1,
            max_tokens=9000,
            skip_special_tokens=False,
        )
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        self._proc_size = tuple(processor.image_processor.final_size)
        del processor

    def teardown(self) -> None:
        if self.backend == "vllm":
            for attr in ("_llm", "_sampling_params"):
                with contextlib.suppress(AttributeError):
                    delattr(self, attr)
        else:
            for attr in ("_model", "_tokenizer", "_processor", "_gen_config"):
                with contextlib.suppress(AttributeError):
                    delattr(self, attr)
        torch.cuda.empty_cache()

    # -- inference --

    @torch.inference_mode()
    def _infer_batch_hf(self, images: list[Image.Image]) -> list[str]:
        if not images:
            return []
        inputs = self._processor(
            images=images,
            text=[self.task_prompt] * len(images),
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self._device)
        outputs = self._model.generate(**inputs, generation_config=self._gen_config)
        return self._processor.batch_decode(outputs, skip_special_tokens=True)

    def _reset_vllm(self) -> None:
        """Teardown and reinit vLLM engine (mirrors Cosmos Curate's _reset pattern)."""
        logger.warning("[vLLM] Resetting engine after inference failure")
        with contextlib.suppress(Exception):
            del self._llm
            del self._sampling_params
            torch.cuda.empty_cache()
        self._setup_vllm()

    def _infer_vllm(self, images: list[Image.Image]) -> list[str]:
        if not images:
            return []
        prompts = [{"prompt": self.task_prompt, "multi_modal_data": {"image": img}} for img in images]

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                outputs = self._llm.generate(prompts, self._sampling_params)
                return [output.outputs[0].text for output in outputs]
            except Exception as e:
                logger.warning(f"[vLLM] Inference failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    self._reset_vllm()
                else:
                    raise
        return []

    def _infer_hf(self, images: list[Image.Image]) -> list[str]:
        all_outputs: list[str] = []
        for start in range(0, len(images), self.inference_batch_size):
            batch = images[start : start + self.inference_batch_size]
            try:
                all_outputs.extend(self._infer_batch_hf(batch))
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning(f"Batch inference failed for pages {start}-{start + len(batch) - 1}: {e}")
                all_outputs.extend(self._infer_hf_single_fallback(batch))
        return all_outputs

    def _infer_hf_single_fallback(self, images: list[Image.Image]) -> list[str]:
        """Process each image individually when batch inference fails."""
        results: list[str] = []
        for img in images:
            try:
                results.extend(self._infer_batch_hf([img]))
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning(f"Single page fallback failed: {e}")
                results.append("")
        return results

    # -- process --

    def process(self, task: InterleavedBatch) -> InterleavedBatch | None:
        task_df = task.to_pandas()
        images = []
        for idx, b in enumerate(task_df["binary_content"]):
            try:
                images.append(Image.open(io.BytesIO(b)))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Skipping page {idx} in {task.task_id}: {e}")
                images.append(None)

        valid_mask = [img is not None for img in images]
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            return None

        valid_outputs = self._infer_vllm(valid_images) if self.backend == "vllm" else self._infer_hf(valid_images)

        all_outputs = []
        valid_iter = iter(valid_outputs)
        for is_valid in valid_mask:
            all_outputs.append(next(valid_iter) if is_valid else "")

        task_df["text_content"] = all_outputs

        metadata = dict(task._metadata)
        metadata["proc_size"] = list(self._proc_size)
        metadata["model_path"] = self.model_path

        return InterleavedBatch(
            task_id=f"{task.task_id}_inferred",
            dataset_name=task.dataset_name,
            data=pa.Table.from_pandas(task_df, preserve_index=False),
            _metadata=metadata,
            _stage_perf=task._stage_perf,
        )
