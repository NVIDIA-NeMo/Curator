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

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.tasks import ImageBatch, ImageObject


class TestImageEmbeddingStage:
    """Test suite for ImageEmbeddingStage."""

    @pytest.fixture
    def stage(self) -> ImageEmbeddingStage:
        """Create a test stage instance."""
        return ImageEmbeddingStage(
            model_dir="test_models/clip",
            model_inference_batch_size=2,
            verbose=True
        )

    @pytest.fixture
    def mock_model(self) -> Mock:
        """Create a mock CLIP model."""
        model = Mock()
        model.setup = Mock()
        # Mock to return embeddings of size 512
        model.return_value = torch.randn(4, 512)
        return model

    @pytest.fixture
    def sample_image_objects(self) -> list[ImageObject]:
        """Create sample ImageObject instances with image data."""
        rng = np.random.default_rng(42)  # Use new RNG with seed for reproducibility
        return [
            ImageObject(
                image_id="img_001",
                image_path="/path/to/img1.jpg",
                image_data=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            ),
            ImageObject(
                image_id="img_002",
                image_path="/path/to/img2.jpg",
                image_data=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            ),
            ImageObject(
                image_id="img_003",
                image_path="/path/to/img3.jpg",
                image_data=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            ),
            ImageObject(
                image_id="img_004",
                image_path="/path/to/img4.jpg",
                image_data=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            )
        ]

    @pytest.fixture
    def sample_image_batch(self, sample_image_objects: list[ImageObject]) -> ImageBatch:
        """Create a sample ImageBatch."""
        return ImageBatch(
            data=sample_image_objects,
            dataset_name="test_dataset",
            task_id="test_task_001",
            _metadata={"test": "metadata"},
            _stage_perf={}
        )

    def test_stage_properties(self, stage: ImageEmbeddingStage) -> None:
        """Test stage properties."""
        assert stage.name == "image_embedding"
        # Allow either requesting GPUs or not, depending on environment
        assert stage.resources.gpus in (0.25, 0.0)
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

    @patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_setup(self, mock_clip_embeddings: Mock, stage: ImageEmbeddingStage) -> None:
        """Test stage setup."""
        mock_model = Mock()
        mock_clip_embeddings.return_value = mock_model

        stage.setup()

        mock_clip_embeddings.assert_called_once()
        call_args, call_kwargs = mock_clip_embeddings.call_args
        assert (
            (len(call_args) >= 1 and call_args[0] == "test_models/clip")
            or (call_kwargs.get("model_dir") == "test_models/clip")
        )
        mock_model.setup.assert_called_once()
        assert stage.model == mock_model

    @patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    @patch("transformers.CLIPProcessor.from_pretrained")
    def test_process_embedding_generation(
        self,
        mock_processor: Mock,
        mock_clip_embeddings: Mock,
        stage: ImageEmbeddingStage,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test the main processing and embedding generation logic."""
        mock_clip_embeddings.return_value = mock_model

        # Mock the processor
        mock_processor_instance = Mock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(4, 3, 224, 224)}
        mock_processor.return_value = mock_processor_instance

        stage.setup()

        # Mock the model to return specific embeddings for each batch call
        embedding_dim = 512
        # The stage processes in batches of 2, so we'll have 2 calls
        # First call: batch 0-2 (2 images), Second call: batch 2-4 (2 images)
        batch1_embeddings = torch.ones(2, embedding_dim) * 1.0  # First batch gets 1s
        batch2_embeddings = torch.ones(2, embedding_dim) * 2.0  # Second batch gets 2s
        mock_model.side_effect = [batch1_embeddings, batch2_embeddings]

        result = stage.process(sample_image_batch)

        # Check that all images have embeddings assigned
        assert len(result.data) == 4
        for img_obj in result.data:
            assert hasattr(img_obj, "embedding")
            assert img_obj.embedding is not None
            assert img_obj.embedding.shape == (embedding_dim,)

        # Check that embeddings were assigned (first 2 should be 1s, last 2 should be 2s)
        expected_values = [1.0, 1.0, 2.0, 2.0]
        for i, img_obj in enumerate(result.data):
            expected_embedding = np.ones(embedding_dim) * expected_values[i]
            np.testing.assert_array_equal(img_obj.embedding, expected_embedding)

        # Check that original task is returned (not a new one)
        assert result is sample_image_batch

    @patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    @patch("transformers.CLIPProcessor.from_pretrained")
    def test_batch_processing(
        self, mock_processor: Mock, mock_clip_embeddings: Mock, stage: ImageEmbeddingStage, mock_model: Mock
    ) -> None:
        """Test that large batches are processed in smaller chunks."""
        # Create stage with model_inference_batch_size=2
        stage = ImageEmbeddingStage(model_inference_batch_size=2)
        mock_clip_embeddings.return_value = mock_model

        # Mock the processor
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance

        stage.setup()

        # Create 5 images (should be processed in 3 batches: 2, 2, 1)
        rng = np.random.default_rng(42)
        images = []
        for i in range(5):
            images.append(ImageObject(
                image_id=f"img_{i:03d}",
                image_path=f"/path/to/img{i}.jpg",
                image_data=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            ))

        batch = ImageBatch(data=images, task_id="test_batch", dataset_name="test_dataset")

        # Mock processor to return appropriate tensor sizes
        # The processor returns the same structure regardless of input size
        mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}

        # Mock model to return embeddings - will be called multiple times
        mock_model.return_value = torch.randn(2, 512)  # Return 2 embeddings per call

        result = stage.process(batch)

        # Should call model multiple times for batches
        assert mock_model.call_count >= 1
        # All 5 images should have embeddings
        assert len(result.data) == 5
        for img_obj in result.data:
            assert hasattr(img_obj, "embedding")
            assert img_obj.embedding.shape == (512,)

    @patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_empty_batch(self, mock_clip_embeddings: Mock, stage: ImageEmbeddingStage) -> None:
        """Test processing empty image batch."""
        empty_batch = ImageBatch(data=[], task_id="empty_test", dataset_name="test_dataset")
        mock_clip_embeddings.return_value = Mock()

        stage.setup()

        result = stage.process(empty_batch)

        assert len(result.data) == 0
        # Model should not be called for empty batch
        stage.model.assert_not_called()

    @patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    @patch("nemo_curator.stages.image.embedders.clip_embedder.logger")
    def test_verbose_logging(
        self,
        mock_logger: Mock,
        mock_clip_embeddings: Mock,
        stage: ImageEmbeddingStage,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test verbose logging output."""
        mock_clip_embeddings.return_value = mock_model

        stage.setup()
        mock_model.return_value = torch.randn(4, 512)

        stage.process(sample_image_batch)

        # Should log embedding generation
        embedding_calls = [call for call in mock_logger.info.call_args_list
                          if "Generated embeddings for" in str(call)]
        assert len(embedding_calls) > 0

    @patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    @patch("transformers.CLIPProcessor.from_pretrained")
    def test_preserves_other_image_attributes(
        self, mock_processor: Mock, mock_clip_embeddings: Mock, stage: ImageEmbeddingStage, sample_image_batch: ImageBatch
    ) -> None:
        """Test that processing preserves other image attributes."""
        mock_clip_embeddings.return_value = Mock()
        mock_clip_embeddings.return_value.return_value = torch.randn(4, 512)

        # Mock the processor
        mock_processor_instance = Mock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(4, 3, 224, 224)}
        mock_processor.return_value = mock_processor_instance

        stage.setup()

        # Add some additional attributes to test preservation
        sample_image_batch.data[0].custom_attr = "test_value"
        sample_image_batch.data[0].metadata = {"caption": "test caption"}

        result = stage.process(sample_image_batch)

        # Check that other attributes are preserved
        assert hasattr(result.data[0], "custom_attr")
        assert result.data[0].custom_attr == "test_value"
        assert hasattr(result.data[0], "metadata")
        assert result.data[0].metadata == {"caption": "test caption"}

    @patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    @patch("transformers.CLIPProcessor.from_pretrained")
    def test_different_batch_sizes(
        self, mock_processor: Mock, mock_clip_embeddings: Mock, sample_image_batch: ImageBatch
    ) -> None:
        """Test embedding generation with different batch sizes."""
        mock_clip_embeddings.return_value = Mock()

        # Mock the processor
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance

        # Test with model_inference_batch_size=1
        small_stage = ImageEmbeddingStage(model_inference_batch_size=1)
        small_stage.setup()

        # Test with model_inference_batch_size=10 (larger than input)
        large_stage = ImageEmbeddingStage(model_inference_batch_size=10)
        large_stage.setup()

        # Mock processor and model returns for different batch sizes
        mock_processor_instance.return_value = {"pixel_values": torch.randn(4, 3, 224, 224)}

        # For small stage (model_inference_batch_size=1), it will be called 4 times with 1 embedding each
        small_stage.model.return_value = torch.randn(1, 512)
        # For large stage (model_inference_batch_size=10), it will be called 1 time with 4 embeddings
        large_stage.model.return_value = torch.randn(4, 512)

        # Process with small batches (should call model 4 times)
        small_stage.model.reset_mock()
        small_result = small_stage.process(sample_image_batch)
        assert small_stage.model.call_count == 4

        # Process with large batch (should call model 1 time)
        large_stage.model.reset_mock()
        large_result = large_stage.process(sample_image_batch)
        assert large_stage.model.call_count == 1

        # Both should produce embeddings for all images
        assert len(small_result.data) == 4
        assert len(large_result.data) == 4

    @patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_processor_integration(
        self, mock_clip_embeddings: Mock, stage: ImageEmbeddingStage, sample_image_batch: ImageBatch
    ) -> None:
        """Test integration with CLIPImageEmbeddings model."""
        # Mock the CLIPImageEmbeddings model to return fixed embeddings
        mock_model_instance = Mock()
        # Return fixed embeddings for consistent testing
        mock_model_instance.return_value = torch.ones(2, 512)  # model_inference_batch_size=2, embedding_dim=512
        mock_clip_embeddings.return_value = mock_model_instance

        stage.setup()
        result = stage.process(sample_image_batch)

        # Verify the model was instantiated and setup was called
        mock_clip_embeddings.assert_called_once()
        call_args, call_kwargs = mock_clip_embeddings.call_args
        assert (
            (len(call_args) >= 1 and call_args[0] == "test_models/clip")
            or (call_kwargs.get("model_dir") == "test_models/clip")
        )
        mock_model_instance.setup.assert_called_once()

        # Verify the model was called twice (for 2 batches of 2 images each)
        assert mock_model_instance.call_count == 2

        # Verify all images have embeddings
        assert all(img.embedding is not None for img in result.data)

    def test_embedding_shape_consistency(self, stage: ImageEmbeddingStage) -> None:
        """Test that embeddings have consistent shape across different inputs."""
        with (
            patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings") as mock_clip_embeddings,
            patch("transformers.CLIPProcessor.from_pretrained") as mock_processor,
        ):
            mock_clip_embeddings.return_value = Mock()

            # Mock the processor
            mock_processor_instance = Mock()
            mock_processor.return_value = mock_processor_instance

            stage.setup()

            # Test different image sizes
            rng = np.random.default_rng(42)
            different_sized_images = [
                ImageObject(
                    image_id="small_img",
                    image_path="/path/to/small.jpg",
                    image_data=rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
                ),
                ImageObject(
                    image_id="large_img",
                    image_path="/path/to/large.jpg",
                    image_data=rng.integers(0, 255, (500, 500, 3), dtype=np.uint8)
                )
            ]

            batch = ImageBatch(data=different_sized_images, task_id="shape_test", dataset_name="test_dataset")

            # Mock consistent outputs regardless of input size
            mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
            stage.model.return_value = torch.randn(2, 512)

            result = stage.process(batch)

            # All embeddings should have the same shape
            for img_obj in result.data:
                assert img_obj.embedding.shape == (512,)

    # GPU integration test with a dummy CUDA-backed embedding model
    @pytest.mark.gpu
    def test_image_embedding_stage_on_gpu(self) -> None:
        if not torch.cuda.is_available():  # pragma: no cover - CPU CI
            pytest.skip("CUDA not available; skipping GPU embedding test")

        class _DummyCLIPImageEmbeddings:
            def __init__(self, _model_dir: str | None = None) -> None:
                pass

            def setup(self) -> None:
                return None

            def __call__(self, batch_numpy: np.ndarray | list[np.ndarray]) -> torch.Tensor:
                device = torch.device("cuda")
                # Stage passes a list of numpy arrays; accept both list and ndarray
                if isinstance(batch_numpy, list):
                    batch_numpy = np.stack(batch_numpy, axis=0)
                x = torch.from_numpy(batch_numpy).to(device=device, dtype=torch.float32)
                if x.ndim == 3:
                    x = x.unsqueeze(0)
                # Compute a simple scalar per image and expand to 16-d embedding
                s = x.mean(dim=(1, 2, 3))  # (N,)
                return s.unsqueeze(1).repeat(1, 16)  # (N, 16)

        rng = np.random.default_rng(123)
        import tempfile

        tmp_dir = tempfile.gettempdir()
        images = [
            ImageObject(
                image_id=f"img_{i:03d}",
                image_path=f"{tmp_dir}/img_{i:03d}.jpg",
                image_data=rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
            )
            for i in range(4)
        ]
        batch = ImageBatch(data=images, dataset_name="ds", task_id="t0")

        stage = ImageEmbeddingStage(model_dir="/does/not/matter", model_inference_batch_size=2, verbose=False)
        with patch(
            "nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings",
            _DummyCLIPImageEmbeddings,
        ):
            stage.setup()
            out = stage.process(batch)

        for img in out.data:
            assert img.embedding is not None
            assert isinstance(img.embedding, np.ndarray)
            assert img.embedding.shape == (16,)

    @patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_remove_image_data_when_enabled(
        self,
        mock_clip_embeddings: Mock,
        sample_image_batch: ImageBatch,
    ) -> None:
        """When remove_image_data=True, image_data should be cleared after processing."""
        stage = ImageEmbeddingStage(model_inference_batch_size=2, remove_image_data=True)

        mock_model_instance = Mock()
        mock_model_instance.setup = Mock()
        mock_model_instance.return_value = torch.randn(2, 512)
        mock_clip_embeddings.return_value = mock_model_instance

        stage.setup()
        result = stage.process(sample_image_batch)

        # Embeddings should be set
        assert all(getattr(img, "embedding", None) is not None for img in result.data)
        # Image data should be removed
        assert all(img.image_data is None for img in result.data)

    @patch("nemo_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_preserve_image_data_when_disabled(
        self,
        mock_clip_embeddings: Mock,
        sample_image_batch: ImageBatch,
    ) -> None:
        """When remove_image_data=False, image_data should remain intact after processing."""
        stage = ImageEmbeddingStage(model_inference_batch_size=2, remove_image_data=False)

        # Keep references to original arrays to verify they are preserved
        original_arrays = [img.image_data for img in sample_image_batch.data]

        mock_model_instance = Mock()
        mock_model_instance.setup = Mock()
        mock_model_instance.return_value = torch.randn(2, 512)
        mock_clip_embeddings.return_value = mock_model_instance

        stage.setup()
        result = stage.process(sample_image_batch)

        # Embeddings should be set
        assert all(getattr(img, "embedding", None) is not None for img in result.data)
        # Image data should be preserved
        assert all(img.image_data is not None for img in result.data)
        # Optionally, verify identity preservation (no replacement)
        assert all(img.image_data is original_arrays[i] for i, img in enumerate(result.data))
