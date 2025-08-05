import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from ray_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage

from ray_curator.tasks import ImageBatch, ImageObject
from ray_curator.backends.base import WorkerMetadata


class TestImageNSFWFilterStage:
    """Test suite for ImageNSFWFilterStage."""

    @pytest.fixture
    def stage(self):
        """Create a test stage instance."""
        return ImageNSFWFilterStage(
            model_dir="test_models/nsfw",
            score_threshold=0.5,
            batch_size=2,
            verbose=True
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock NSFW model."""
        model = Mock()
        model.setup = Mock()
        model.return_value = torch.tensor([[0.3], [0.7], [0.1], [0.9]])  # Mock scores
        return model

    @pytest.fixture
    def sample_image_objects(self):
        """Create sample ImageObject instances with embeddings."""
        return [
            ImageObject(
                image_id="img_001",
                image_path="/path/to/img1.jpg",
                embedding=np.random.rand(512).astype(np.float32),
                image_data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            ),
            ImageObject(
                image_id="img_002", 
                image_path="/path/to/img2.jpg",
                embedding=np.random.rand(512).astype(np.float32),
                image_data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            ),
            ImageObject(
                image_id="img_003",
                image_path="/path/to/img3.jpg", 
                embedding=np.random.rand(512).astype(np.float32),
                image_data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            ),
            ImageObject(
                image_id="img_004",
                image_path="/path/to/img4.jpg",
                embedding=np.random.rand(512).astype(np.float32),
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
        assert stage.name == "image_nsfw_filter"
        assert stage.resources.gpus == 0.25
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

    @patch('ray_curator.stages.image.filters.nsfw_filter.NSFWScorer')
    def test_setup(self, mock_nsfw_scorer, stage):
        """Test stage setup."""
        mock_model = Mock()
        mock_nsfw_scorer.return_value = mock_model
        
        stage.setup()
        
        mock_nsfw_scorer.assert_called_once_with(model_dir="test_models/nsfw")
        mock_model.setup.assert_called_once()
        assert stage.model == mock_model

    @patch('ray_curator.stages.image.filters.nsfw_filter.NSFWScorer')
    def test_process_filtering(self, mock_nsfw_scorer, stage, sample_image_batch, mock_model):
        """Test the main processing and filtering logic."""
        mock_nsfw_scorer.return_value = mock_model
        stage.setup()
        
        # Mock the model to return specific scores
        # Stage processes in batches of 2, so we have 2 batch calls
        # Batch 1: img_001: 0.3 (keep), img_002: 0.7 (filter)
        # Batch 2: img_003: 0.1 (keep), img_004: 0.9 (filter)
        batch1_scores = torch.tensor([[0.3], [0.7]])
        batch2_scores = torch.tensor([[0.1], [0.9]])
        mock_model.side_effect = [batch1_scores, batch2_scores]
        
        result = stage.process(sample_image_batch)
        
        # Should keep 2 images (scores 0.3 and 0.1 < threshold 0.5)
        assert len(result.data) == 2
        assert result.data[0].image_id == "img_001"
        assert result.data[1].image_id == "img_003"
        
        # Check scores were assigned correctly (use approximate comparison for floats)
        assert abs(result.data[0].nsfw_score - 0.3) < 1e-6
        assert abs(result.data[1].nsfw_score - 0.1) < 1e-6
        
        # Check metadata preservation
        assert result.dataset_name == "test_dataset"
        assert result.task_id == "test_task_001_image_nsfw_filter"
        assert result._metadata == {"test": "metadata"}

    @patch('ray_curator.stages.image.filters.nsfw_filter.NSFWScorer')
    def test_threshold_boundary_cases(self, mock_nsfw_scorer, stage, sample_image_batch, mock_model):
        """Test filtering behavior at threshold boundaries."""
        mock_nsfw_scorer.return_value = mock_model
        stage.setup()
        
        # Test with scores exactly at threshold (0.5)
        # Stage processes in batches of 2, so we need to return results for each batch call
        batch1_scores = torch.tensor([[0.49], [0.5]])     # First batch: keep first, filter second
        batch2_scores = torch.tensor([[0.51], [0.499]])   # Second batch: filter first, keep second
        mock_model.side_effect = [batch1_scores, batch2_scores]
        
        result = stage.process(sample_image_batch)
        
        # Should keep images with scores < 0.5 (0.49 and 0.499)
        assert len(result.data) == 2
        # Check that all kept images have scores < 0.5
        for img_obj in result.data:
            assert img_obj.nsfw_score < 0.5

    @patch('ray_curator.stages.image.filters.nsfw_filter.NSFWScorer')
    def test_all_images_filtered(self, mock_nsfw_scorer, stage, sample_image_batch, mock_model):
        """Test case where all images are filtered out."""
        mock_nsfw_scorer.return_value = mock_model
        stage.setup()
        
        # All scores above threshold
        mock_model.return_value = torch.tensor([[0.8], [0.9], [0.7], [0.6]])
        
        result = stage.process(sample_image_batch)
        
        assert len(result.data) == 0

    @patch('ray_curator.stages.image.filters.nsfw_filter.NSFWScorer')
    def test_no_images_filtered(self, mock_nsfw_scorer, stage, sample_image_batch, mock_model):
        """Test case where no images are filtered out."""
        mock_nsfw_scorer.return_value = mock_model
        stage.setup()
        
        # All scores below threshold
        mock_model.return_value = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
        
        result = stage.process(sample_image_batch)
        
        assert len(result.data) == 4

    @patch('ray_curator.stages.image.filters.nsfw_filter.NSFWScorer')
    def test_batch_processing(self, mock_nsfw_scorer, stage, mock_model):
        """Test that large batches are processed in smaller chunks."""
        # Create stage with batch_size=2
        stage = ImageNSFWFilterStage(batch_size=2, score_threshold=0.5)
        mock_nsfw_scorer.return_value = mock_model
        stage.setup()
        
        # Create 5 images (should be processed in 3 batches: 2, 2, 1)
        images = []
        for i in range(5):
            images.append(ImageObject(
                image_id=f"img_{i:03d}",
                image_path=f"/path/to/img{i}.jpg",
                embedding=np.random.rand(512).astype(np.float32)
            ))
        
        batch = ImageBatch(data=images, task_id="test_batch", dataset_name="test_dataset")
        
        # Mock model to return scores in the order they're processed
        mock_model.side_effect = [
            torch.tensor([[0.1], [0.2]]),  # First batch: 2 images
            torch.tensor([[0.3], [0.4]]),  # Second batch: 2 images  
            torch.tensor([[0.6]])          # Third batch: 1 image
        ]
        
        result = stage.process(batch)
        
        # Should call model 3 times (for 3 batches)
        assert mock_model.call_count == 3
        # Should keep 4 images (all except the last one with score 0.6)
        assert len(result.data) == 4

    def test_different_thresholds(self, sample_image_batch):
        """Test filtering with different threshold values."""
        # Test with very strict threshold (0.1)
        strict_stage = ImageNSFWFilterStage(score_threshold=0.1)
        
        # Test with very lenient threshold (0.9)  
        lenient_stage = ImageNSFWFilterStage(score_threshold=0.9)
        
        # Mock setup for both stages
        with patch('ray_curator.stages.image.filters.nsfw_filter.NSFWScorer') as mock_nsfw_scorer:
            mock_model = Mock()
            # Default batch_size is 32, so all 4 images will be processed in one batch
            all_scores = torch.tensor([[0.05], [0.2], [0.5], [0.95]])  # Scores: 0.05, 0.2, 0.5, 0.95
            mock_model.side_effect = [all_scores, all_scores]  # Same scores for both stages
            mock_nsfw_scorer.return_value = mock_model
            
            strict_stage.setup()
            lenient_stage.setup()
            
            strict_result = strict_stage.process(sample_image_batch)
            lenient_result = lenient_stage.process(sample_image_batch)
            
            # Strict threshold should only keep the first image (0.05 < 0.1)
            assert len(strict_result.data) == 1
            
            # Lenient threshold should keep first 3 images (0.05, 0.2, 0.5 < 0.9, 0.95 filtered)
            assert len(lenient_result.data) == 3

    @patch('ray_curator.stages.image.filters.nsfw_filter.NSFWScorer')
    @patch('ray_curator.stages.image.filters.nsfw_filter.logger')
    def test_verbose_logging(self, mock_logger, mock_nsfw_scorer, stage, sample_image_batch, mock_model):
        """Test verbose logging output."""
        mock_nsfw_scorer.return_value = mock_model
        stage.setup()
        
        # Mock scores: 2 keep, 2 filter
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.1], [0.9]])
        
        result = stage.process(sample_image_batch)
        
        # Should log for filtered images only
        filtered_calls = [call for call in mock_logger.info.call_args_list 
                         if "filtered out as NSFW" in str(call)]
        assert len(filtered_calls) == 2  # 2 images filtered
        
        # Should log summary
        summary_calls = [call for call in mock_logger.info.call_args_list 
                        if "NSFW filtering:" in str(call)]
        assert len(summary_calls) == 1

    def test_empty_batch(self, stage):
        """Test processing empty image batch."""
        empty_batch = ImageBatch(data=[], task_id="empty_test", dataset_name="test_dataset")
        
        with patch('ray_curator.stages.image.filters.nsfw_filter.NSFWScorer') as mock_nsfw_scorer:
            mock_model = Mock()
            mock_nsfw_scorer.return_value = mock_model
            stage.setup()
            
            result = stage.process(empty_batch)
            
            assert len(result.data) == 0
            # Model should not be called for empty batch
            mock_model.assert_not_called()
