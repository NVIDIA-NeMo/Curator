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
DEFAULT_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"


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
    task_prompt
        Prompt string sent to the model for each page image.
    backend
        Inference backend: ``"vllm"`` or ``"hf"``.
    inference_batch_size
        Pages per GPU forward pass (HF backend only).
    max_num_seqs
        Maximum concurrent sequences (vLLM backend only).
    """

    model_path: str = DEFAULT_MODEL_PATH
    task_prompt: str = DEFAULT_TASK_PROMPT
    backend: str = "vllm"
    inference_batch_size: int = 4
    max_num_seqs: int = 64
    name: str = "nemotron_parse_inference"
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0, gpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    # -- setup / teardown --

    def setup(self, worker_metadata: dict | None = None) -> None:  # noqa: ARG002
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
        from vllm import LLM, SamplingParams

        resolved_path = self._resolve_local_model_path()
        logger.info(f"[vLLM] Loading {resolved_path} with max_num_seqs={self.max_num_seqs}")
        self._llm = LLM(
            model=resolved_path,
            max_num_seqs=self.max_num_seqs,
            limit_mm_per_prompt={"image": 1},
            dtype="bfloat16",
            trust_remote_code=True,
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
        logger.info(f"[vLLM] Model loaded, proc_size={self._proc_size}")

    def _resolve_local_model_path(self) -> str:
        """Resolve the HF model to a local snapshot path to avoid repeated API calls."""
        from huggingface_hub import snapshot_download

        return snapshot_download(self.model_path, local_files_only=True)

    def teardown(self) -> None:
        if self.backend == "vllm":
            del self._llm, self._sampling_params
        else:
            del self._model, self._tokenizer, self._processor, self._gen_config
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

    def _infer_vllm(self, images: list[Image.Image]) -> list[str]:
        if not images:
            return []
        prompts = [{"prompt": self.task_prompt, "multi_modal_data": {"image": img}} for img in images]
        outputs = self._llm.generate(prompts, self._sampling_params)
        return [output.outputs[0].text for output in outputs]

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
            except (RuntimeError, ValueError, TypeError) as e:  # noqa: PERF203
                logger.warning(f"Single page fallback failed: {e}")
                results.append("")
        return results

    # -- process --

    def process(self, task: InterleavedBatch) -> InterleavedBatch | None:
        task_df = task.to_pandas()
        images = [Image.open(io.BytesIO(b)) for b in task_df["binary_content"]]

        all_outputs = self._infer_vllm(images) if self.backend == "vllm" else self._infer_hf(images)

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
