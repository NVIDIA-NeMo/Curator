import pathlib
import re
from typing import Any

from loguru import logger

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    # Create dummy classes for type hints when vllm is not available
    class LLM:
        pass

    class SamplingParams:
        pass


from ray_curator.models.base import ModelInterface
from ray_curator.utils import grouping

_QWEN2_5_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_QWEN_VARIANTS_INFO = {
    "qwen": _QWEN2_5_VL_MODEL_ID,
}


class QwenVL(ModelInterface):
    def __init__(  # noqa: PLR0913
        self,
        model_dir: str,
        model_variant: str,
        caption_batch_size: int,
        fp8: bool = True,
        max_output_tokens: int = 512,
        model_does_preprocess: bool = False,
        disable_mmcache: bool = False,
        stage2_prompt_text: str | None = None,
        verbose: bool = False,
    ):
        self.model_dir = model_dir
        self.model_variant = model_variant
        self.caption_batch_size = caption_batch_size
        self.fp8 = fp8
        self.max_output_tokens = max_output_tokens
        self.model_does_preprocess = model_does_preprocess
        self.disable_mmcache = disable_mmcache
        self.stage2_prompt = stage2_prompt_text
        self.verbose = verbose
        self.weight_file = str(pathlib.Path(model_dir) / _QWEN_VARIANTS_INFO[model_variant])
        # Default pattern for stage2 caption generation - matches (.*)(user_prompt)(.*)
        self.pattern = r"(.*)(user_prompt)(.*)"

    @property
    def model_id_names(self) -> list[str]:
        return [_QWEN_VARIANTS_INFO[self.model_variant]]

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vllm is required for QwenVL model but is not installed. Please install vllm: pip install vllm"
            raise ImportError(msg)

        mm_processor_kwargs = {
            "do_resize": self.model_does_preprocess,
            "do_rescale": self.model_does_preprocess,
            "do_normalize": self.model_does_preprocess,
        }
        self.model = LLM(
            model=self.weight_file,
            limit_mm_per_prompt={"image": 0, "video": 1},
            quantization="fp8" if self.fp8 else None,
            max_seq_len_to_capture=32768,
            max_model_len=32768,
            gpu_memory_utilization=0.85,
            mm_processor_kwargs=mm_processor_kwargs,
            disable_mm_preprocessor_cache=self.disable_mmcache,
            max_num_batched_tokens=32768,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=self.max_output_tokens,
            stop_token_ids=[],
        )
        logger.info(
            "CUDA graph enabled for sequences smaller than 16k tokens; adjust accordingly for even longer sequences"
        )

    def generate(
        self, videos: list[dict[str, Any]], generate_stage2_caption: bool = False, batch_size: int = 16
    ) -> list[str]:
        generated_text = []
        for batch_videos in grouping.split_by_chunk_size(videos, batch_size):
            model_inputs = list(batch_videos)
            try:
                outputs = self.model.generate(
                    model_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

                if generate_stage2_caption:
                    for i, out in enumerate(outputs):
                        out_caption = out.outputs[0].text
                        updated_prompt = self.stage2_prompt + out_caption
                        model_inputs[i]["prompt"] = re.sub(
                            self.pattern,
                            rf"\1{updated_prompt}\3",
                            model_inputs[i]["prompt"],
                            flags=re.DOTALL,
                        )
                    outputs = self.model.generate(
                        model_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )
                generated_text.extend(out.outputs[0].text for out in outputs)
            except Exception as e:
                logger.error(f"Error generating caption for batch: {e}")
                raise
        return generated_text
