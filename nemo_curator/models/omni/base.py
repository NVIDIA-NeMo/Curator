"""Base model classes for VLM inference."""

from abc import ABC, abstractmethod
import base64
import concurrent.futures
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Iterable, TypeVar, Generator, Sequence
import functools
import tempfile
import os

from PIL import Image

from loguru import logger


T = TypeVar('T')


@dataclass(kw_only=True)
class InferenceConfig:
    """Configuration for inference."""

    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = False
    priority_mode: bool = False


@dataclass(kw_only=True)
class ModelConfig:
    """Model-specific configuration."""

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_tokens: int | None = None
    
    def __post_init__(self) -> None:
        assert self.gpu_memory_utilization >= 0.0 and self.gpu_memory_utilization <= 1.0, "GPU memory utilization must be between 0.0 and 1.0"
        assert self.tensor_parallel_size >= 1, "Tensor parallel size must be greater than 0"


class Model(ABC):
    """Abstract base class for models."""

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return False

    @abstractmethod
    def load(self) -> None:
        """Load the model and processor."""
        ...
    
    @abstractmethod
    def preload(self) -> None:
        """Preload the model into the local disk cache."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        self._model = None


class VLMModel(Model):
    """Abstract base class for vision-language models."""

    def __init__(self, model_id: str, model_config: ModelConfig) -> None:
        """Initialize VLM model.

        Args:
            model_id: HuggingFace model identifier.
        """
        self.model_id = model_id
        self._model: Any = None
        self.model_config = model_config

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @abstractmethod
    def load(self) -> None:
        """Load the model and processor."""
        ...
    
    @abstractmethod
    def preload(self) -> None:
        """Preload the model into the local disk cache."""
        ...

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image] | None,
        inference_config: InferenceConfig,
    ) -> list[str]:
        """Generate responses for a batch of prompts, optionally with images.

        Args:
            prompts: List of text prompts.
            images: List of PIL images corresponding to prompts, or None for text-only.
            inference_config: Inference configuration.

        Returns:
            List of generated response strings.
        """
        ...

    def unload(self) -> None:
        """Unload model and free resources."""
        self._model = None


@dataclass(kw_only=True)
class VLLMModelConfig(ModelConfig):
    """vLLM-specific configuration."""

    enforce_eager: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class VLLMModel(VLMModel):
    """Base class for vLLM-based vision-language models."""

    model_config: VLLMModelConfig

    def __init__(
        self,
        model_id: str,
        model_config: VLLMModelConfig,
    ) -> None:
        """Initialize vLLM model.

        Args:
            model_id: HuggingFace model identifier.
            vllm_config: vLLM-specific configuration.
        """
        super().__init__(model_id, model_config)

    def load(self) -> None:
        """Load the vLLM model."""
        if self.is_loaded:
            return
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="vllm-torchinductor-cache-")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_OFFLINE = True

        from huggingface_hub import try_to_load_from_cache

        model_path = try_to_load_from_cache(self.model_id, "config.json").rsplit("/", 1)[0]

        from vllm import LLM

        self._model = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.model_config.tensor_parallel_size,
            gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            enforce_eager=self.model_config.enforce_eager,
            max_model_len=self.model_config.max_tokens,
            **self.model_config.extra_kwargs,
        )
        self._tokenizer = self._model.get_tokenizer()
    
    def preload(self) -> None:
        """Preload the model into the local disk cache."""
        assert not self.is_loaded, "Model is already loaded"
        
        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_OFFLINE = False
        from huggingface_hub import try_to_load_from_cache

        if try_to_load_from_cache(self.model_id, "config.json") is not None:
            logger.info(f"Model {self.model_id} already in local cache")
        else:
            from huggingface_hub import snapshot_download
            logger.info(f"Model {self.model_id} not in local cache, downloading...")
            snapshot_download(self.model_id)
            if try_to_load_from_cache(self.model_id, "config.json") is not None:
                logger.info(f"Model {self.model_id} downloaded and cached")
            else:
                raise Exception(f"Failed to preload model {self.model_id}")

    def precompute_embeddings(self, images: list[Image.Image]) -> list[Any]:
        """Precompute embeddings for a list of images."""
        self._model.precompute_embeddings(images)
        return [None for _ in images]
    
    def _get_input_dict(self, prompt: str, image: Image.Image | None) -> dict[str, Any]:
        if image is not None:
            content = [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt
        messages = [{"role": "user", "content": content}]
        formatted_prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_dict: dict[str, Any] = {"prompt": formatted_prompt}
        if image is not None:
            input_dict["multi_modal_data"] = {"image": image}
        return input_dict

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image] | None,
        inference_config: InferenceConfig,
    ) -> list[str]:
        """Generate responses using vLLM.

        Args:
            prompts: List of text prompts.
            images: List of PIL images corresponding to prompts, or None for text-only.
            inference_config: Inference configuration.

        Returns:
            List of generated response strings.
        """
        from vllm import SamplingParams

        vllm_inputs = []
        for i, prompt in enumerate(prompts):
            image = images[i] if images else None
            vllm_inputs.append(self._get_input_dict(prompt, image))

        sampling_params = SamplingParams(
            max_tokens=self.model_config.max_tokens,
            temperature=inference_config.temperature,
            top_p=inference_config.top_p,
            repetition_penalty=inference_config.repetition_penalty,
        )

        outputs = self._model.generate(vllm_inputs, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]

    def generate_stream(
        self,
        prompts: Iterable[tuple[str, Image.Image, Callable[[str | None, int | None], T | None]]],
        inference_config: InferenceConfig,
        step: Callable[[], None] | None = None,
        max_parallel_tasks: int = 16,
    ) -> Generator[T, None, None]:
        """Generate responses using vLLM and stream the output.

        Args:
            prompts: Generator of tuples of text prompts, images, and processor functions.
                The prompt can be None to skip a sample.
                The processor function has the arguments (text: str | None, token_count: int | None).
                It is called for each token block generated by the model.
                For the last token block (also when skipped), it is called with (None, None).
            inference_config: Inference configuration.
            max_parallel_tasks: Maximum number of parallel tasks.
            step: Callback function to update progress after each inference step.

        Returns:
            Generator of task results.
        """

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.model_config.max_tokens,
            temperature=inference_config.temperature,
            top_p=inference_config.top_p,
            repetition_penalty=inference_config.repetition_penalty,
        )

        pending_task_ids = []

        pending_task_processors = {}
        pending_task_results = {}

        prompt_iter = iter(prompts)

        assert not self._model.llm_engine.has_unfinished_requests(), "Model still has unfinished requests"

        def add_next_task():
            """Add the next task, return True if successful/done, False if skipped."""
            try:
                prompt, image, next_text_fn = next(prompt_iter)
            except StopIteration:
                return True
            
            request_id = str(next(self._model.request_counter))

            if prompt is None:
                # Skip sample
                pending_task_ids.append(request_id)
                pending_task_results[request_id] = next_text_fn(None, None)
                return False

            input_dict = self._get_input_dict(prompt, image)
            
            self._model.llm_engine.add_request(request_id, input_dict, sampling_params)
            pending_task_ids.append(request_id)
            pending_task_processors[request_id] = next_text_fn
            return True
        
        def _flush_tasks():
            """Yield results of finished tasks in-order."""
            while len(pending_task_ids) > 0 and pending_task_ids[0] in pending_task_results:
                yield pending_task_results.pop(pending_task_ids.pop(0))

        def _skip_add_next_task():
            """Add the next task, yield skipped tasks if at the front of the queue."""
            while not add_next_task():
                # If there is a finished task in the front of the queue, yield it's result.
                yield from _flush_tasks()

        for i in range(max_parallel_tasks):
            yield from _skip_add_next_task()

        while self._model.llm_engine.has_unfinished_requests():
            step_outputs = self._model.llm_engine.step()
            # Process outputs of unfinished tasks.
            for output in step_outputs:
                assert output.request_id in pending_task_processors, f"Unexpected request ID {output.request_id}"
                next_text_fn = pending_task_processors[output.request_id]
                try:
                    result = next_text_fn(output.outputs[0].text, len(output.outputs[0].token_ids))
                    if result is not None:
                        pending_task_results[output.request_id] = result
                        self._model.llm_engine.abort_request([output.request_id])
                        yield from _skip_add_next_task()
                    if output.finished:
                        pending_task_results[output.request_id] = next_text_fn(None, None)
                        yield from _skip_add_next_task()
                except Exception as e:
                    self._model.llm_engine.abort_request([output.request_id])
                    pending_task_results[output.request_id] = e
                    yield from _skip_add_next_task()
            yield from _flush_tasks()
            if step is not None:
                step()


@dataclass(kw_only=True)
class TransformersModelConfig(ModelConfig):
    """Transformers-specific configuration."""

    device_map: str = "auto"
    torch_dtype: str = "auto"
    use_flash_attention: bool = True


class TransformersModel(VLMModel):
    """Base class for transformers-based vision-language models."""

    model_config: TransformersModelConfig

    def __init__(
        self,
        model_id: str,
        model_config: TransformersModelConfig,
    ) -> None:
        """Initialize transformers model.

        Args:
            model_id: HuggingFace model identifier.
            model_config: Transformers-specific configuration.
        """
        super().__init__(model_id, model_config)

    def _get_max_memory(self) -> dict[int | str, str] | None:
        """Calculate max_memory dict based on gpu_memory_utilization.

        Returns:
            Dict mapping device IDs to memory limits, or None for no limit.
        """
        if self.model_config.gpu_memory_utilization >= 1.0:
            return None

        import torch

        if not torch.cuda.is_available():
            return None

        max_memory = {}
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            max_bytes = int(total_mem * self.model_config.gpu_memory_utilization)
            max_memory[i] = f"{max_bytes // (1024**3)}GiB"

        return max_memory

    def preload(self) -> None:
        """Preload the model into the local disk cache."""
        from huggingface_hub import try_to_load_from_cache

        if try_to_load_from_cache(self.model_id, "config.json") is not None:
            logger.info(f"Model {self.model_id} already in local cache")
        else:
            from huggingface_hub import snapshot_download

            logger.info(f"Model {self.model_id} not in local cache, downloading...")
            snapshot_download(self.model_id)
            if try_to_load_from_cache(self.model_id, "config.json") is not None:
                logger.info(f"Model {self.model_id} downloaded and cached")
            else:
                raise Exception(f"Failed to preload model {self.model_id}")



T = TypeVar('T', bound=Callable[[], Any])

def with_hf_offline_mode(fn: T) -> T:
    """Decorator to enable or disable offline mode for HuggingFace.

    Args:
        fn: Function to decorate.
    """
    import huggingface_hub.constants as hf_constants

    def wrapper(*args, **kwargs) -> Any:
        original_offline = hf_constants.HF_HUB_OFFLINE
        hf_constants.HF_HUB_OFFLINE = True
        try:
            return fn(*args, **kwargs)
        except Exception:
            hf_constants.HF_HUB_OFFLINE = original_offline
            return fn(*args, **kwargs)
        finally:
            hf_constants.HF_HUB_OFFLINE = original_offline
    return wrapper


class TaggingModel(Model):
    """Base class for image tagging models."""

    @abstractmethod
    def get_tags(
        self,
        images: list[Image.Image],
        threshold: float = 0.5,
    ) -> list[list[str]]:
        """Return tags with confidence scores for each image."""


@dataclass(kw_only=True)
class SGLangModelConfig(ModelConfig):
    """SGLang-specific configuration."""

    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class SGLangModel(VLMModel):
    """Base class for SGLang-based vision-language models."""

    model_config: SGLangModelConfig

    def __init__(
        self,
        model_id: str,
        model_config: SGLangModelConfig,
    ) -> None:
        """Initialize SGLang model.

        Args:
            model_id: HuggingFace model identifier.
            model_config: SGLang-specific configuration.
        """
        super().__init__(model_id, model_config)

    def load(self) -> None:
        """Load the SGLang model."""
        if self.is_loaded:
            return

        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_OFFLINE = True

        import sglang as sgl

        from huggingface_hub import try_to_load_from_cache

        model_path = try_to_load_from_cache(self.model_id, "config.json").rsplit("/", 1)[0]

        self._model = sgl.Engine(
            model_path=model_path,
            tp_size=self.model_config.tensor_parallel_size,
            enable_multimodal=True,
            mem_fraction_static=self.model_config.gpu_memory_utilization,
            **self.model_config.extra_kwargs,
        )

    def preload(self) -> None:
        """Preload the model into the local disk cache."""
        assert not self.is_loaded, "Model is already loaded"

        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_OFFLINE = False
        from huggingface_hub import try_to_load_from_cache

        if try_to_load_from_cache(self.model_id, "config.json") is not None:
            logger.info(f"Model {self.model_id} already in local cache")
        else:
            from huggingface_hub import snapshot_download
            logger.info(f"Model {self.model_id} not in local cache, downloading...")
            snapshot_download(self.model_id)
            if try_to_load_from_cache(self.model_id, "config.json") is not None:
                logger.info(f"Model {self.model_id} downloaded and cached")
            else:
                raise Exception(f"Failed to preload model {self.model_id}")

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image] | None,
        inference_config: InferenceConfig,
    ) -> list[str]:
        """Generate responses using SGLang.

        Args:
            prompts: List of text prompts.
            images: List of PIL images corresponding to prompts, or None for text-only.
            inference_config: Inference configuration.

        Returns:
            List of generated response strings.
        """
        tokenizer = self._model.get_tokenizer()

        sgl_inputs = []
        for i, prompt in enumerate(prompts):
            image = images[i] if images else None
            content: list[dict[str, Any]] = []
            if image is not None:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            input_dict: dict[str, Any] = {"text": formatted_prompt}
            if image is not None:
                input_dict["image_data"] = image
            sgl_inputs.append(input_dict)

        sampling_params = {
            "max_new_tokens": self.model_config.max_tokens,
            "temperature": inference_config.temperature,
            "top_p": inference_config.top_p,
            "repetition_penalty": inference_config.repetition_penalty,
        }

        outputs = self._model.generate(sgl_inputs, sampling_params=sampling_params)
        return [output["text"] for output in outputs]


@dataclass(kw_only=True)
class NVInferenceModelConfig(ModelConfig):
    """NVIDIA Inference API-specific configuration."""

    base_url: str = "https://inference-api.nvidia.com"
    api_key_env_var: str = "NVINFERENCE_API_KEY"

    def __post_init__(self) -> None:
        pass


class NVInferenceModel(VLMModel):
    """Base class for models using NVIDIA Inference API."""

    model_config: NVInferenceModelConfig

    def __init__(
        self,
        model_id: str,
        model_config: NVInferenceModelConfig,
    ) -> None:
        """Initialize NVIDIA Inference API model.

        Args:
            model_id: Model identifier for NVIDIA Inference API.
            model_config: NVIDIA Inference API-specific configuration.
        """
        super().__init__(model_id, model_config)
        self._client: Any = None

    @property
    def is_loaded(self) -> bool:
        """Check if client is initialized."""
        return self._client is not None

    def load(self) -> None:
        """Initialize the NVIDIA Inference API client."""
        if self.is_loaded:
            return

        from nemo_curator.models.client.nvinference_client import (
            create_openai_client,
            get_nvinference_api_key,
        )

        logger.info(f"Initializing NVIDIA Inference client for model {self.model_id}")
        api_key = get_nvinference_api_key(self.model_config.api_key_env_var)
        self._client = create_openai_client(
            api_key=api_key, base_url=self.model_config.base_url
        )
        logger.info("NVIDIA Inference client initialized")

        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.model_config.tensor_parallel_size)

    def preload(self) -> None:
        """Preload is not needed for API-based models."""
        logger.info(f"Preload not required for API-based model {self.model_id}")

    def unload(self) -> None:
        """Unload client and free resources."""
        self._client = None
        logger.info("NVIDIA Inference client unloaded")

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string.

        Args:
            image: PIL Image to encode.

        Returns:
            Base64-encoded image string.
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _build_message_content(
        self, prompt: str, image: Image.Image | str | None
    ) -> list[dict[str, Any]]:
        """Build message content for OpenAI-compatible API.

        Args:
            prompt: Text prompt.
            image: Optional PIL Image or URI string.

        Returns:
            List of content dictionaries.
        """
        content: list[dict[str, Any]] = []

        content.append({"type": "text", "text": prompt})
        if image is not None:
            if isinstance(image, str):
                image_url = image
            else:
                image_b64 = self._encode_image_to_base64(image)
                image_url = f"data:image/png;base64,{image_b64}"
            
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        return content

    def _build_message_content_legacy(
        self, prompt: str, image: Image.Image | str | None
    ) -> list[dict[str, Any]]:
        """Build message content for OpenAI-compatible API (legacy format).

        Args:
            prompt: Text prompt.
            image: Optional PIL Image or URI string (legacy format).

        Returns:
            List of content dictionaries (legacy format).
        """
        content: list[dict[str, Any]] = []

        if image is not None:
            if isinstance(image, str):
                image_url = image
            else:
                image_b64 = self._encode_image_to_base64(image)
                image_url = f"data:image/png;base64,{image_b64}"
            image_url = f"<image>{image_url}</image>"
        else:
            image_url = ""

        content.append({"role": "user", "content": f"{image_url}\n{prompt}"})
        return content

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image | str] | None,
        inference_config: InferenceConfig,
    ) -> list[str]:
        """Generate responses using NVIDIA Inference API.

        Args:
            prompts: List of text prompts.
            images: List of PIL images or URI strings corresponding to prompts, or None for text-only.
            inference_config: Inference configuration.

        Returns:
            List of generated response strings.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from nemo_curator.models.client.nvinference_client import stream_chat_completion_text

        results = []
        for i, prompt in enumerate(prompts):
            image = images[i] if images else None
            content = self._build_message_content(prompt, image)
            messages = [{"role": "user", "content": content}]

            extra_headers = {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"} if inference_config.priority_mode else None

            try:
                response = stream_chat_completion_text(
                    self._client,
                    model=self.model_id,
                    messages=messages,
                    temperature=inference_config.temperature,
                    top_p=inference_config.top_p,
                    max_tokens=self.model_config.max_tokens or 2048,
                    extra_headers=extra_headers,
                )
                results.append(response if response else "")
            except Exception as e:
                logger.error(f"Error generating response for prompt {i}: {e}")
                results.append("")

        return results

    def generate_stream(
        self,
        prompts: Iterable[tuple[str | None, Image.Image | str | None, Callable[[str | None, int | None], T | None]]],
        inference_config: InferenceConfig,
        step: Callable[[], None] | None = None,
        max_parallel_tasks: int = 16,
    ) -> Generator[T, None, None]:
        """Generate responses using vLLM and stream the output.

        Args:
            prompts: Generator of tuples of text prompts, images, and processor functions.
            inference_config: Inference configuration.
            max_parallel_tasks: Maximum number of parallel tasks.
            step: Callback function to update progress after each inference step.

        Returns:
            Generator of task results.
        """
        cancelled = False

        import httpx
        import openai
        import backoff

        @backoff.on_exception(
            functools.partial(backoff.expo, max_value=10),
            (openai.APIError, httpx.HTTPError),
            max_time=300,
        )
        def run_inference(prompt: str | None, image: Image.Image | str | None, processor: Callable[[str | None, int | None], T | None]) -> T | None:
            if prompt is None:
                # Skipped sample
                return processor(None, None)
            content = self._build_message_content(prompt, image)
            messages = [{"role": "user", "content": content}]

            if cancelled:
                raise concurrent.futures.CancelledError()

            extra_headers = {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"} if inference_config.priority_mode else None

            completion = None
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=inference_config.temperature,
                    top_p=inference_config.top_p,
                    max_tokens=self.model_config.max_tokens,
                    stream=True,
                    extra_headers=extra_headers,
                )
                accumulated_content = ""
                total_tokens = 0
                for chunk in completion:
                    if cancelled:
                        raise concurrent.futures.CancelledError()
                    # print("Chunk:",chunk)
                    if chunk.usage:
                        total_tokens = chunk.usage.completion_tokens
                    elif chunk.choices[0].delta.content is not None:
                        total_tokens = (len(accumulated_content) + len(chunk.choices[0].delta.content)) // 4
                    if chunk.choices[0].delta.content is not None:
                        accumulated_content += chunk.choices[0].delta.content
                        result = processor(accumulated_content, total_tokens)
                        if result is not None:
                            return result
                    step()
                # print("Completion:", completion)
                return processor(None, None)
            except (openai.APIError, httpx.HTTPError) as e:
                print(f"APIError: {e!r}, retrying")
                raise
            except Exception as e:
                print(f"Fatal Error: {e!r}, crashing")
                import traceback
                traceback.print_exc()
                raise
            finally:
                if completion is not None:
                    completion.close()

        pending_steps_ordered = []
        pending_steps = set()

        prompt_iter = iter(prompts)

        def add_next_step():
            try:
                prompt, image, processor = next(prompt_iter)
            except StopIteration:
                return
            future = executor.submit(run_inference, prompt, image, processor)
            pending_steps_ordered.append(future)
            pending_steps.add(future)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            try:
                for _ in range(max_parallel_tasks):
                    add_next_step()

                while len(pending_steps) > 0:
                    done, pending_steps = concurrent.futures.wait(pending_steps, return_when=concurrent.futures.FIRST_COMPLETED)
                    for _ in done:
                        # Add a new step to the pool for each done step
                        add_next_step()
                    # Yield in order of submission
                    while pending_steps_ordered:
                        if pending_steps_ordered[0].done():
                            yield pending_steps_ordered.pop(0).result()
            except Exception as e:
                print(f"Fatal Error: {e!r}, crashing now")
                import traceback
                traceback.print_exc()
                raise
            finally:
                cancelled = True
