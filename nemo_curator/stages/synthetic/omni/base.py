"""Base stage class extending NeMo Curator's ProcessingStage."""

import math
import os
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from loguru import logger
from PIL import Image

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.omni.base import InferenceConfig, VLMModel
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources, _get_gpu_memory_gb
from nemo_curator.tasks.image import SingleDataTask

T = TypeVar("T")


class SkipSample(Exception):  # noqa: N818
    """Exception to be raised in build_prompt to skip a sample."""


class VLMProcessingStage(ProcessingStage[SingleDataTask[T], SingleDataTask[T]], Generic[T]):
    """Base class for VLM SDG pipeline stages.

    Extends NeMo Curator's ProcessingStage for image captioning tasks.
    All custom stages should inherit from this class.
    """

    name: str = "vlm_base_stage"
    resources: Resources = Resources(cpus=1.0)

    def __init__(self, *, cuda_devices: Sequence[int] | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self.cuda_devices = cuda_devices
        if cuda_devices is not None:
            if len(cuda_devices) != self.resources.gpus:
                msg = f"cuda_devices length must match resources.gpus={self.resources.gpus}, got {len(cuda_devices)}"
                raise ValueError(msg)
        elif self.resources.gpus > 0:
            # None means use runtime-assigned GPU(s) (e.g. Ray sets CUDA_VISIBLE_DEVICES per worker).
            pass

    def _maybe_set_cuda_device(self) -> None:
        """Set the current CUDA device for this stage, if configured."""
        if self.cuda_devices is None or len(self.cuda_devices) == 0:
            return
        import torch

        if not torch.cuda.is_available():
            msg = "CUDA is not available"
            raise RuntimeError(msg)
        torch.cuda.set_device(self.cuda_devices[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(f"{d}" for d in self.cuda_devices)

    def _get_gpu_memory_utilization(self) -> float:
        """Get the GPU memory utilization for the stage."""
        available_memory = _get_gpu_memory_gb()
        if self.resources.gpus > 0:
            return 0.9
        else:
            if self.resources.gpu_memory_gb is None:
                msg = "GPU memory GB must be set"
                raise ValueError(msg)
            if self.resources.gpu_memory_gb <= 0:
                msg = "GPU memory GB must be greater than 0"
                raise ValueError(msg)
            if self.resources.gpu_memory_gb > available_memory:
                msg = "GPU memory GB must be less than available memory"
                raise ValueError(msg)
            return self.resources.gpu_memory_gb / available_memory

    def _get_tensor_parallel_size(self) -> int:
        """Get tensor parallel size based on resource allocation."""
        return math.ceil(self.resources.gpus) if self.resources.gpus > 0 else 1

    def setup(self, worker_metadata: dict | None = None) -> None:
        """Initialize models or allocate resources.

        Override this method to load models, warm up caches, etc.

        Args:
            worker_metadata: Optional metadata from the worker.
        """

    def process(self, task: SingleDataTask) -> SingleDataTask | list[SingleDataTask]:
        """Process a single task.

        Override this method to implement stage-specific processing logic.

        Args:
            task: Input SingleDataTask.

        Returns:
            Processed SingleDataTask or list of tasks.
        """
        return task

    def teardown(self) -> None:
        """Clean up resources after processing.

        Override this method to unload models, free memory, etc.
        """


class ModelProcessingStage(VLMProcessingStage[T], Generic[T]):
    """Base class for stages using VLMModel for inference.

    Provides common model loading, generation, and teardown logic.
    Subclasses should implement `build_prompt` and `handle_response`.
    """

    name: str = "model_base_stage"
    resources: Resources = Resources(cpus=8.0)
    batch_size: int = 8
    multimodal: bool = True

    def __init__(
        self,
        model: VLMModel,
        inference_config: InferenceConfig,
        batch_size: int = 8,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize model processing stage.

        Args:
            model: VLMModel instance for inference.
            inference_config: Configuration for inference.
            batch_size: Number of items to process per batch.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self.model = model
        self.inference_config = inference_config
        self.batch_size = batch_size

    def _initialize_model(self) -> None:
        """Initialize the model."""
        self.model.load()

    def setup_on_node(self, _node_info: NodeInfo, _worker_metadata: WorkerMetadata) -> None:
        # TODO: Add weight downloading logic
        self._initialize_model()

    def setup(self, _worker_metadata: dict | None = None) -> None:
        """Load the model."""
        if self.model.is_loaded:
            return
        self._initialize_model()

    @abstractmethod
    def build_prompt(self, task: SingleDataTask[T]) -> str:
        """Build the text prompt for a task.

        Args:
            task: The task to build a prompt for.

        Returns:
            The text prompt string.

        Raises:
            SkipSample: If the sample should be skipped (i.e. does not set the is_valid flag, just skips).
            Exception: If any other error occurs, sets the is_valid flag to False and sets the error message of the sample.
        """
        ...

    @abstractmethod
    def handle_response(self, task: SingleDataTask[T], response: str) -> SingleDataTask[T]:
        """Handle the model response for a task.

        Args:
            task: The task to update.
            response: The model's generated text.

        Raises:
            Exception: If any other error occurs, sets the is_valid flag to False and sets the error message of the sample.
        """
        ...

    def load_image(self, task: SingleDataTask[T]) -> Image.Image:
        """Load image from task. Override for custom loading logic.

        Args:
            task: The task containing image information.

        Returns:
            PIL Image object.
        """
        from nemo_curator.stages.synthetic.omni.io import load_image_from_task

        return load_image_from_task(task)

    def process(self, task: SingleDataTask[T]) -> SingleDataTask[T]:
        """Process a single task.

        Args:
            task: Input task.

        Returns:
            Processed task.
        """
        return self.process_batch([task])[0]

    def _handle_response_one(self, tasks: list[SingleDataTask[T]], idx: int, response: str) -> None:
        """Call handle_response for one task, catching and logging errors."""
        try:
            self.handle_response(tasks[idx], response)
        except SkipSample:
            logger.debug(f"{self.name}: skipping sample {idx}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"{self.name}: error handling response for task {idx}: {e}")
            tasks[idx].data.error = f"{self.name}: {e}"
            tasks[idx].data.is_valid = False

    def process_batch(self, tasks: list[SingleDataTask[T]]) -> list[SingleDataTask[T]]:
        """Process a batch of tasks.

        Skips tasks that already failed in previous stages (is_valid=False).

        Args:
            tasks: List of tasks to process.

        Returns:
            List of processed tasks.
        """
        if not tasks:
            return []

        valid_indices: list[int] = []
        prompts: list[str] = []
        images: list[Image.Image] = []

        for i, task in enumerate(tasks):
            # Skip tasks that already failed in previous stages
            if not task.data.is_valid:
                logger.debug(f"{self.name}: skipping invalid task {i}")
                continue

            try:
                image = self.load_image(task) if self.multimodal else None
                prompt = self.build_prompt(task)
                valid_indices.append(i)
                prompts.append(prompt)
                images.append(image)
            except SkipSample:
                logger.debug(f"{self.name}: skipping sample {i}")
                continue
            except Exception as e:  # noqa: BLE001
                logger.error(f"{self.name}: error preparing task {i}: {e}")
                task.data.error = f"{self.name}: {e}"
                task.data.is_valid = False

        if not valid_indices:
            return tasks

        try:
            responses = self.model.generate(prompts, images if self.multimodal else None, self.inference_config)

            for idx, response in zip(valid_indices, responses, strict=False):
                self._handle_response_one(tasks, idx, response)

            logger.info(f"{self.name}: processed batch of {len(valid_indices)} items")

        except Exception as e:  # noqa: BLE001
            logger.error(f"{self.name}: batch processing error: {e}")
            for task in tasks:
                task.data.error = f"{self.name}: {e}"
                task.data.is_valid = False

        return tasks

    def preload_model(self) -> None:
        """Preload the model."""
        self.model.preload()

    def teardown(self) -> None:
        """Unload model."""
        self.model.unload()
