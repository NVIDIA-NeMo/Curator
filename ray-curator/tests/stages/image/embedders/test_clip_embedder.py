import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from ray_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from ray_curator.tasks import ImageBatch, ImageObject
from ray_curator.backends.base import WorkerMetadata


class TestImageEmbeddingStage:
    """Test suite for ImageEmbeddingStage."""

    @pytest.fixture
    def stage(self):
        """Create a test stage instance."""
        return ImageEmbeddingStage(
            model_dir="test_models/clip",
            batch_size=2,
            verbose=True
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock CLIP model."""
        model = Mock()
        model.setup = Mock()
        # Mock to return embeddings of size 512
        model.return_value = torch.randn(4, 512)
        return model

    @pytest.fixture
    def sample_image_objects(self):
        """Create sample ImageObject instances with image data."""
        return [
            ImageObject(
                image_id="img_001",
                image_path="/path/to/img1.jpg",
                image_data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            ),
            ImageObject(
                image_id="img_002", 
                image_path="/path/to/img2.jpg",
                image_data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            ),
            ImageObject(
                image_id="img_003",
                image_path="/path/to/img3.jpg", 
                image_data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            ),
            ImageObject(
                image_id="img_004",
                image_path="/path/to/img4.jpg",
                image_data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
        ]

    @pytest.fixture
    def sample_image_batch(self, sample_image_objects):
        """Create a sample ImageBatch."""
        return ImageBatch(
            data=sample_image_objects,
            dataset_name="test_dataset",
            task_id="test_task_001",
            _metadata={"test": "metadata"},
            _stage_perf={}
        )

    def test_stage_properties(self, stage):
        """Test stage properties."""
        assert stage.name == "image_embedding"
        assert stage.resources.gpus == 0.25
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

    @patch('ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings')
    def test_setup(self, mock_clip_embeddings, stage):
        """Test stage setup."""
        mock_model = Mock()
        mock_clip_embeddings.return_value = mock_model
        
        stage.setup()
        
        mock_clip_embeddings.assert_called_once_with(model_dir="test_models/clip")
        mock_model.setup.assert_called_once()
        assert stage.model == mock_model

    @patch('ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings')
    @patch('transformers.CLIPProcessor.from_pretrained')
    def test_process_embedding_generation(self, mock_processor, mock_clip_embeddings, stage, sample_image_batch, mock_model):
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
        for i, img_obj in enumerate(result.data):
            assert hasattr(img_obj, 'embedding')
            assert img_obj.embedding is not None
            assert img_obj.embedding.shape == (embedding_dim,)
            
        # Check that embeddings were assigned (first 2 should be 1s, last 2 should be 2s)
        expected_values = [1.0, 1.0, 2.0, 2.0]
        for i, img_obj in enumerate(result.data):
            expected_embedding = np.ones(embedding_dim) * expected_values[i]
            np.testing.assert_array_equal(img_obj.embedding, expected_embedding)
        
        # Check that original task is returned (not a new one)
        assert result is sample_image_batch

    @patch('ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings')
    @patch('transformers.CLIPProcessor.from_pretrained')
    def test_batch_processing(self, mock_processor, mock_clip_embeddings, stage, mock_model):
        """Test that large batches are processed in smaller chunks."""
        # Create stage with batch_size=2
        stage = ImageEmbeddingStage(batch_size=2)
        mock_clip_embeddings.return_value = mock_model
        
        # Mock the processor
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        stage.setup()
        
        # Create 5 images (should be processed in 3 batches: 2, 2, 1)
        images = []
        for i in range(5):
            images.append(ImageObject(
                image_id=f"img_{i:03d}",
                image_path=f"/path/to/img{i}.jpg",
                image_data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
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
            assert hasattr(img_obj, 'embedding')
            assert img_obj.embedding.shape == (512,)

    @patch('ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings')
    @patch('transformers.CLIPProcessor.from_pretrained')
    def test_empty_batch(self, mock_processor, mock_clip_embeddings, stage):
        """Test processing empty image batch."""
        empty_batch = ImageBatch(data=[], task_id="empty_test", dataset_name="test_dataset")
        mock_clip_embeddings.return_value = Mock()
        
        stage.setup()
        
        result = stage.process(empty_batch)
        
        assert len(result.data) == 0
        # Model should not be called for empty batch
        stage.model.assert_not_called()

    @patch('ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings')
    @patch('transformers.CLIPProcessor.from_pretrained')
    @patch('ray_curator.stages.image.embedders.clip_embedder.logger')
    def test_verbose_logging(self, mock_logger, mock_processor, mock_clip_embeddings, stage, sample_image_batch, mock_model):
        """Test verbose logging output."""
        mock_clip_embeddings.return_value = mock_model
        
        # Mock the processor
        mock_processor_instance = Mock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(4, 3, 224, 224)}
        mock_processor.return_value = mock_processor_instance
        
        stage.setup()
        mock_model.return_value = torch.randn(4, 512)
        
        result = stage.process(sample_image_batch)
        
        # Should log embedding generation
        embedding_calls = [call for call in mock_logger.info.call_args_list 
                          if "Generated embeddings for" in str(call)]
        assert len(embedding_calls) > 0

    @patch('ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings')
    @patch('transformers.CLIPProcessor.from_pretrained')
    def test_preserves_other_image_attributes(self, mock_processor, mock_clip_embeddings, stage, sample_image_batch):
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
        assert hasattr(result.data[0], 'custom_attr')
        assert result.data[0].custom_attr == "test_value"
        assert hasattr(result.data[0], 'metadata')
        assert result.data[0].metadata == {"caption": "test caption"}

    @patch('ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings')
    @patch('transformers.CLIPProcessor.from_pretrained')
    def test_different_batch_sizes(self, mock_processor, mock_clip_embeddings, sample_image_batch):
        """Test embedding generation with different batch sizes."""
        mock_clip_embeddings.return_value = Mock()
        
        # Mock the processor
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        # Test with batch_size=1
        small_stage = ImageEmbeddingStage(batch_size=1)
        small_stage.setup()
        
        # Test with batch_size=10 (larger than input)
        large_stage = ImageEmbeddingStage(batch_size=10)
        large_stage.setup()
        
        # Mock processor and model returns for different batch sizes
        mock_processor_instance.return_value = {"pixel_values": torch.randn(4, 3, 224, 224)}
        
        # For small stage (batch_size=1), it will be called 4 times with 1 embedding each
        small_stage.model.return_value = torch.randn(1, 512)
        # For large stage (batch_size=10), it will be called 1 time with 4 embeddings
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

    @patch('ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings')
    @patch('transformers.CLIPProcessor.from_pretrained')
    def test_processor_integration(self, mock_processor, mock_clip_embeddings, stage, sample_image_batch):
        """Test integration with CLIPProcessor."""
        mock_clip_embeddings.return_value = Mock()
        mock_clip_embeddings.return_value.return_value = torch.randn(4, 512)
        
        # Mock the processor to verify it's called correctly
        mock_processor_instance = Mock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(4, 3, 224, 224)}
        mock_processor.return_value = mock_processor_instance
        
        stage.setup()
        
        result = stage.process(sample_image_batch)
        
        # Verify processor was called (it gets called for each batch)
        # With batch_size=2 and 4 images, we expect 2 calls to from_pretrained
        assert mock_processor.call_count >= 1
        # Check that it was called with the correct model name
        mock_processor.assert_any_call("openai/clip-vit-large-patch14")
        
        # Verify processor instance was called with PIL Images
        assert mock_processor_instance.call_count > 0
        call_args = mock_processor_instance.call_args
        assert 'images' in call_args[1]
        assert 'return_tensors' in call_args[1]
        assert call_args[1]['return_tensors'] == 'pt'

    def test_embedding_shape_consistency(self, stage):
        """Test that embeddings have consistent shape across different inputs."""
        with patch('ray_curator.stages.image.embedders.clip_embedder.CLIPImageEmbeddings') as mock_clip_embeddings:
            with patch('transformers.CLIPProcessor.from_pretrained') as mock_processor:
                mock_clip_embeddings.return_value = Mock()
                
                # Mock the processor
                mock_processor_instance = Mock()
                mock_processor.return_value = mock_processor_instance
                
                stage.setup()
                
                # Test different image sizes
                different_sized_images = [
                    ImageObject(
                        image_id="small_img",
                        image_path="/path/to/small.jpg",
                        image_data=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    ),
                    ImageObject(
                        image_id="large_img", 
                        image_path="/path/to/large.jpg",
                        image_data=np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
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
