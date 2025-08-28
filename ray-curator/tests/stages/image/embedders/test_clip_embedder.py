from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from ray_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from ray_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from ray_curator.tasks import DocumentBatch, ImageBatch, ImageObject


@pytest.fixture
def sample_image_objects() -> list[ImageObject]:
    """Create sample ImageObject instances with image data (module-scoped for reuse)."""
    rng = np.random.default_rng(42)
    return [
        ImageObject(
            image_id="img_001",
            image_path="/path/to/img1.jpg",
            image_data=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        ),
        ImageObject(
            image_id="img_002",
            image_path="/path/to/img2.jpg",
            image_data=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        ),
        ImageObject(
            image_id="img_003",
            image_path="/path/to/img3.jpg",
            image_data=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        ),
        ImageObject(
            image_id="img_004",
            image_path="/path/to/img4.jpg",
            image_data=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        ),
    ]


@pytest.fixture
def sample_image_batch(sample_image_objects: list[ImageObject]) -> ImageBatch:
    """Create a sample ImageBatch (module-scoped for reuse)."""
    return ImageBatch(
        data=sample_image_objects,
        dataset_name="test_dataset",
        task_id="test_task_001",
        _metadata={"test": "metadata"},
        _stage_perf={},
    )


class TestImageEmbeddingStage:
    """Test suite for ImageEmbeddingStage."""

    @pytest.fixture
    def stage(self) -> ImageEmbeddingStage:
        """Create a test stage instance."""
        return ImageEmbeddingStage(
            model_dir="test_models/clip",
            model_inference_batch_size=2,
            verbose=True,
        )

    @pytest.fixture
    def mock_model(self) -> Mock:
        """Create a mock CLIP model."""
        model = Mock()
        model.setup = Mock()

        def _side_effect(batch: list[np.ndarray]) -> torch.Tensor:
            batch_size = len(batch)
            return torch.randn(batch_size, 512)

        model.side_effect = _side_effect
        return model

    def test_stage_properties(self, stage: ImageEmbeddingStage) -> None:
        """Test stage properties."""
        assert stage.name == "image_embedding"
        # Allow either requesting GPUs or not, depending on environment
        assert stage.resources.gpus in (0.25, 0.0)
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
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

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_process_embedding_generation(
        self,
        mock_clip_embeddings: Mock,
        stage: ImageEmbeddingStage,
        sample_image_batch: ImageBatch,
    ) -> None:
        """Test the main processing and embedding generation logic."""
        # Two batches of 2; return deterministic values per batch
        embedding_dim = 512
        batch1 = torch.ones(2, embedding_dim) * 1.0
        batch2 = torch.ones(2, embedding_dim) * 2.0

        mock_model = Mock()
        mock_model.setup = Mock()
        mock_model.side_effect = [batch1, batch2]
        mock_clip_embeddings.return_value = mock_model

        stage.setup()
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

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_batch_processing(self, mock_clip_embeddings: Mock) -> None:
        """Test that large batches are processed in smaller chunks."""
        stage = ImageEmbeddingStage(model_inference_batch_size=2)

        mock_model = Mock()
        mock_model.setup = Mock()

        def _side_effect(batch: list[np.ndarray]) -> torch.Tensor:
            return torch.randn(len(batch), 512)

        mock_model.side_effect = _side_effect
        mock_clip_embeddings.return_value = mock_model

        stage.setup()

        # Create 5 images (should be processed in 3 batches: 2, 2, 1)
        rng = np.random.default_rng(42)
        images = []
        for i in range(5):
            images.append(
                ImageObject(
                    image_id=f"img_{i:03d}",
                    image_path=f"/path/to/img{i}.jpg",
                    image_data=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                )
            )

        batch = ImageBatch(data=images, task_id="test_batch", dataset_name="test_dataset")
        result = stage.process(batch)

        # Should call model multiple times for batches
        assert mock_model.call_count == 3
        # All 5 images should have embeddings
        assert len(result.data) == 5
        for img_obj in result.data:
            assert hasattr(img_obj, "embedding")
            assert img_obj.embedding.shape == (512,)

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_empty_batch(self, mock_clip_embeddings: Mock, stage: ImageEmbeddingStage) -> None:
        """Test processing empty image batch."""
        empty_batch = ImageBatch(data=[], task_id="empty_test", dataset_name="test_dataset")
        mock_clip_embeddings.return_value = Mock()

        stage.setup()

        result = stage.process(empty_batch)

        assert len(result.data) == 0
        # Model should not be called for empty batch
        stage.model.assert_not_called()

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    @patch("ray_curator.stages.image.embedders.clip_embedder.logger")
    def test_verbose_logging(
        self,
        mock_logger: Mock,
        mock_clip_embeddings: Mock,
        stage: ImageEmbeddingStage,
        sample_image_batch: ImageBatch,
    ) -> None:
        """Test verbose logging output."""
        mock_model = Mock()
        mock_model.setup = Mock()
        mock_model.side_effect = [torch.randn(2, 512), torch.randn(2, 512)]
        mock_clip_embeddings.return_value = mock_model

        stage.setup()
        stage.process(sample_image_batch)

        # Should log embedding generation
        embedding_calls = [
            call for call in mock_logger.info.call_args_list if "Generated embeddings for" in str(call)
        ]
        assert len(embedding_calls) > 0

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_preserves_other_image_attributes(
        self, mock_clip_embeddings: Mock, stage: ImageEmbeddingStage, sample_image_batch: ImageBatch
    ) -> None:
        """Test that processing preserves other image attributes."""
        mock_model = Mock()
        mock_model.setup = Mock()
        mock_model.side_effect = [torch.randn(2, 512), torch.randn(2, 512)]
        mock_clip_embeddings.return_value = mock_model

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

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_different_batch_sizes(self, mock_clip_embeddings: Mock, sample_image_batch: ImageBatch) -> None:
        """Test embedding generation with different batch sizes."""
        mock_model_small = Mock()
        mock_model_small.setup = Mock()
        mock_model_small.side_effect = [torch.randn(1, 512) for _ in range(4)]

        mock_model_large = Mock()
        mock_model_large.setup = Mock()
        mock_model_large.side_effect = [torch.randn(4, 512)]

        # Test with model_inference_batch_size=1
        mock_clip_embeddings.return_value = mock_model_small
        small_stage = ImageEmbeddingStage(model_inference_batch_size=1)
        small_stage.setup()
        small_result = small_stage.process(sample_image_batch)
        assert mock_model_small.call_count == 4

        # Test with model_inference_batch_size=10 (larger than input)
        mock_clip_embeddings.return_value = mock_model_large
        large_stage = ImageEmbeddingStage(model_inference_batch_size=10)
        large_stage.setup()
        large_result = large_stage.process(sample_image_batch)
        assert mock_model_large.call_count == 1

        # Both should produce embeddings for all images
        assert len(small_result.data) == 4
        assert len(large_result.data) == 4

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_model_integration(
        self, mock_clip_embeddings: Mock, stage: ImageEmbeddingStage, sample_image_batch: ImageBatch
    ) -> None:
        """Test integration with CLIPImageEmbeddings model."""
        # Mock the CLIPImageEmbeddings model to return fixed embeddings
        mock_model_instance = Mock()
        mock_model_instance.setup = Mock()
        mock_model_instance.side_effect = [torch.ones(2, 512), torch.ones(2, 512)]
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
        with patch(
            "ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings"
        ) as mock_clip_embeddings:
            mock_clip_embeddings.return_value = Mock()

            stage.setup()

            # Test different image sizes
            rng = np.random.default_rng(42)
            different_sized_images = [
                ImageObject(
                    image_id="small_img",
                    image_path="/path/to/small.jpg",
                    image_data=rng.integers(0, 255, (100, 100, 3), dtype=np.uint8),
                ),
                ImageObject(
                    image_id="large_img",
                    image_path="/path/to/large.jpg",
                    image_data=rng.integers(0, 255, (500, 500, 3), dtype=np.uint8),
                ),
            ]

            batch = ImageBatch(data=different_sized_images, task_id="shape_test", dataset_name="test_dataset")

            # Mock consistent outputs regardless of input size
            stage.model.side_effect = [torch.randn(2, 512)]

            result = stage.process(batch)

            # All embeddings should have the same shape
            for img_obj in result.data:
                assert img_obj.embedding.shape == (512,)

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_remove_image_data_when_enabled(
        self, mock_clip_embeddings: Mock, sample_image_batch: ImageBatch
    ) -> None:
        """When remove_image_data=True, image_data should be cleared after processing."""
        stage = ImageEmbeddingStage(model_inference_batch_size=2, remove_image_data=True)

        mock_model_instance = Mock()
        mock_model_instance.setup = Mock()
        mock_model_instance.side_effect = [torch.randn(2, 512), torch.randn(2, 512)]
        mock_clip_embeddings.return_value = mock_model_instance

        stage.setup()
        result = stage.process(sample_image_batch)

        # Embeddings should be set
        assert all(getattr(img, "embedding", None) is not None for img in result.data)
        # Image data should be removed
        assert all(img.image_data is None for img in result.data)

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
    def test_preserve_image_data_when_disabled(
        self, mock_clip_embeddings: Mock, sample_image_batch: ImageBatch
    ) -> None:
        """When remove_image_data=False, image_data should remain intact after processing."""
        stage = ImageEmbeddingStage(model_inference_batch_size=2, remove_image_data=False)

        # Keep references to original arrays to verify they are preserved
        original_arrays = [img.image_data for img in sample_image_batch.data]

        mock_model_instance = Mock()
        mock_model_instance.setup = Mock()
        mock_model_instance.side_effect = [torch.randn(2, 512), torch.randn(2, 512)]
        mock_clip_embeddings.return_value = mock_model_instance

        stage.setup()
        result = stage.process(sample_image_batch)

        # Embeddings should be set
        assert all(getattr(img, "embedding", None) is not None for img in result.data)
        # Image data should be preserved
        assert all(img.image_data is not None for img in result.data)
        # Optionally, verify identity preservation (no replacement)
        assert all(img.image_data is original_arrays[i] for i, img in enumerate(result.data))


class TestConvertEmbeddingsToDocumentBatchStage:
    """Tests for converting ImageBatch with embeddings to DocumentBatch."""

    @pytest.fixture
    def convert_stage(self) -> ConvertImageBatchToDocumentBatchStage:
        # Include both image_id and embedding columns
        return ConvertImageBatchToDocumentBatchStage(fields=["image_id", "embedding"])

    @pytest.fixture
    def image_batch_with_embeddings(self) -> ImageBatch:
        rng = np.random.default_rng(7)
        images: list[ImageObject] = []
        for i in range(3):
            embedding = rng.normal(size=(8,)).astype(np.float32)
            import os
            import tempfile
            tmp_dir = os.path.join(tempfile.gettempdir(), "nemo_curator_tests")
            os.makedirs(tmp_dir, exist_ok=True)
            images.append(
                ImageObject(
                    image_id=f"img_{i:03d}",
                    image_path=os.path.join(tmp_dir, f"img_{i:03d}.jpg"),
                    image_data=rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
                    embedding=embedding,
                )
            )
        return ImageBatch(
            data=images,
            dataset_name="ds_test",
            task_id="task_123",
            _metadata={"foo": "bar"},
            _stage_perf={"stage": 1.23},
        )

    def test_conversion_outputs_document_batch(
        self,
        convert_stage: ConvertImageBatchToDocumentBatchStage,
        image_batch_with_embeddings: ImageBatch,
    ) -> None:
        out = convert_stage.process(image_batch_with_embeddings)
        assert isinstance(out, DocumentBatch)

        df = out.to_pandas()
        assert list(df.columns) == ["image_id", "embedding"]
        assert len(df) == 3

        # Validate that image_ids and embeddings are correctly propagated
        src_ids = [img.image_id for img in image_batch_with_embeddings.data]
        out_ids = df["image_id"].tolist()
        assert src_ids == out_ids

        # Compare embeddings row-wise
        for i, img in enumerate(image_batch_with_embeddings.data):
            np.testing.assert_allclose(df.iloc[i]["embedding"], img.embedding)

    def test_metadata_and_ids_are_preserved(
        self,
        convert_stage: ConvertImageBatchToDocumentBatchStage,
        image_batch_with_embeddings: ImageBatch,
    ) -> None:
        out = convert_stage.process(image_batch_with_embeddings)

        # Task id should be suffixed with stage name
        assert out.task_id == f"{image_batch_with_embeddings.task_id}_{convert_stage.name}"
        # Dataset name and metadata should be carried over
        assert out.dataset_name == image_batch_with_embeddings.dataset_name
        assert out._metadata == image_batch_with_embeddings._metadata
        assert out._stage_perf == image_batch_with_embeddings._stage_perf

    def test_empty_input_creates_empty_dataframe(self, convert_stage: ConvertImageBatchToDocumentBatchStage) -> None:
        empty_batch = ImageBatch(data=[], dataset_name="ds", task_id="t0")
        out = convert_stage.process(empty_batch)
        df = out.to_pandas()
        assert isinstance(out, DocumentBatch)
        assert list(df.columns) == ["image_id", "embedding"]
        assert len(df) == 0
    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
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

    @patch("ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings")
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
