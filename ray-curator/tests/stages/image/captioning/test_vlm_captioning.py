from collections.abc import Iterable

import numpy as np
import pytest

from ray_curator.models import ImageBatch, ImageData
from ray_curator.stages.image.captioning.vlm_captioning import VLMCaptioning
from ray_curator.stages.services.model_client import (
    AsyncLLMClient,
    LLMClient,
)


class _DummySyncClient(LLMClient):
    def setup(self) -> None:
        """No-op setup."""
        return

    def query_model(
        self,
        *,
        messages: Iterable,  # noqa: ARG002
        model: str,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> list[str]:
        # Return a deterministic caption for testing
        return ["dummy-caption-sync"]


class _DummyAsyncClient(AsyncLLMClient):
    def setup(self) -> None:
        """No-op setup."""
        return

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,  # noqa: ARG002
        model: str,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> list[str]:
        # Return a deterministic caption for testing
        return ["dummy-caption-async"]


class _ErrorSyncClient(LLMClient):
    def setup(self) -> None:
        """No-op setup."""
        return

    def query_model(
        self,
        *,
        messages: Iterable,  # noqa: ARG002
        model: str,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        error_msg = "invalid API key"
        raise RuntimeError(error_msg)


def _make_image_batch() -> ImageBatch:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    img_obj = ImageData(image_path="test_image.jpg", image_id="img-1", image_data=image)
    return ImageBatch(task_id="batch-1", dataset_name="ds", data=[img_obj])


def test_sync_success_sets_caption():
    batch = _make_image_batch()
    stage = VLMCaptioning(
        client=_DummySyncClient(),
        model_name="model-x",
        prompt="Describe",
        max_tokens=10,
        temperature=0.1,
        verbose=False,
    )
    out = stage.process(batch)
    assert out.data[0].metadata["caption"] == "dummy-caption-sync"


def test_async_success_sets_caption():
    batch = _make_image_batch()
    stage = VLMCaptioning(
        client=_DummyAsyncClient(),
        model_name="model-x",
        prompt="Describe",
        max_tokens=10,
        temperature=0.1,
        verbose=False,
    )
    out = stage.process(batch)
    assert out.data[0].metadata["caption"] == "dummy-caption-async"


def test_sync_error_raises_when_fail_on_error_true():
    batch = _make_image_batch()
    stage = VLMCaptioning(
        client=_ErrorSyncClient(),
        model_name="model-x",
        prompt="Describe",
        fail_on_error=True,
    )
    with pytest.raises(RuntimeError):
        stage.process(batch)
