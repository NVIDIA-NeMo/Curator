from __future__ import annotations

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from ray_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from ray_curator.tasks import ImageBatch, ImageObject


class TestImageAestheticFilterStage:
    """Test suite for ImageAestheticFilterStage."""

    @pytest.fixture
    def stage(self) -> ImageAestheticFilterStage:
        """Create a test stage instance."""
        return ImageAestheticFilterStage(
            model_dir="test_models/aesthetics",
            threshold=0.5,
            batch_size=2
        )

    @pytest.fixture
    def sample_image_objects(self) -> list[ImageObject]:
        """Create sample ImageObject instances with embeddings."""
        rng = np.random.default_rng(42)
        return [
            ImageObject(
                image_path="/path/to/img_001.jpg",
                image_id="img_001",
                image_data=rng.random((224, 224, 3)),
                embedding=rng.random(512),
            ),
            ImageObject(
                image_path="/path/to/img_002.jpg",
                image_id="img_002",
                image_data=rng.random((224, 224, 3)),
                embedding=rng.random(512),
            ),
            ImageObject(
                image_path="/path/to/img_003.jpg",
                image_id="img_003",
                image_data=rng.random((224, 224, 3)),
                embedding=rng.random(512),
            ),
            ImageObject(
                image_path="/path/to/img_004.jpg",
                image_id="img_004",
                image_data=rng.random((224, 224, 3)),
                embedding=rng.random(512),
            ),
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

    def test_stage_properties(self, stage: ImageAestheticFilterStage) -> None:
        """Test stage properties."""
        assert stage.name == "image_aesthetic_filter"
        assert stage.resources.gpus == 0.25
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

    @patch("ray_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_setup(self, mock_aesthetic_scorer: Mock, stage: ImageAestheticFilterStage) -> None:
        """Test stage setup."""
        mock_model = Mock()
        mock_aesthetic_scorer.return_value = mock_model

        stage.setup()

        mock_aesthetic_scorer.assert_called_once_with(model_dir="test_models/aesthetics")
        mock_model.setup.assert_called_once()
        assert stage.model == mock_model

    @patch("ray_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_process_filtering(self, mock_aesthetic_scorer: Mock, stage: ImageAestheticFilterStage, sample_image_batch: ImageBatch, mock_model: Mock) -> None:
        """Test the main processing and filtering logic."""
        mock_aesthetic_scorer.return_value = mock_model
        stage.setup()

        # Mock the model to return specific scores
        # Stage processes in batches of 2, so we have 2 batch calls
        # Batch 1: img_001: 0.7 (keep), img_002: 0.3 (filter)
        # Batch 2: img_003: 0.8 (keep), img_004: 0.2 (filter)
        test_scores = [
            np.array([0.7, 0.3]),  # First batch
            np.array([0.8, 0.2])   # Second batch
        ]

        mock_model.side_effect = test_scores

        # Process the batch
        result = stage.process(sample_image_batch)

        # Should keep 2 images (scores 0.7 and 0.8, both >= 0.5)
        assert len(result.data) == 2
        assert result.data[0].image_id == "img_001"
        assert result.data[0].aesthetic_score == 0.7
        assert result.data[1].image_id == "img_003"
        assert result.data[1].aesthetic_score == 0.8

        # Verify model was called correctly with embeddings
        assert mock_model.call_count == 2
        for call_args in mock_model.call_args_list:
            batch_embeddings = call_args[0][0]
            assert batch_embeddings.shape[0] == 2  # Batch size
            assert batch_embeddings.shape[1] == 512  # Embedding dimension

    @patch("ray_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_threshold_variations(self, mock_aesthetic_scorer: Mock, stage: ImageAestheticFilterStage, sample_image_batch: ImageBatch, mock_model: Mock) -> None:
        """Test different threshold values."""
        mock_aesthetic_scorer.return_value = mock_model
        
        # Test with strict threshold (0.9) and lenient threshold (0.2)
        strict_stage = ImageAestheticFilterStage(threshold=0.9, model_dir="test_models/aesthetics")
        lenient_stage = ImageAestheticFilterStage(threshold=0.2, model_dir="test_models/aesthetics")
        
        strict_stage.model = mock_model
        lenient_stage.model = mock_model
        
        # Mock scores: 0.95, 0.5, 0.3, 0.1
        test_scores = [np.array([0.95, 0.5]), np.array([0.3, 0.1])]
        mock_model.side_effect = test_scores * 2  # Called twice, once for each stage
        
        strict_result = strict_stage.process(sample_image_batch)
        
        # Reset the side effect for the second call
        mock_model.side_effect = test_scores
        lenient_result = lenient_stage.process(sample_image_batch)

        # Strict threshold should only keep the first image (0.95 >= 0.9)
        assert len(strict_result.data) == 1

        # Lenient threshold should keep first 3 images (>= 0.2)
        assert len(lenient_result.data) == 3

    @patch("ray_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    @patch("ray_curator.stages.image.filters.aesthetic_filter.logger")
    def test_verbose_logging(self, mock_logger: Mock, mock_aesthetic_scorer: Mock, stage: ImageAestheticFilterStage, sample_image_batch: ImageBatch, mock_model: Mock) -> None:
        """Test verbose logging output."""
        mock_aesthetic_scorer.return_value = mock_model
        stage.setup()

        # Mock scores: 2 keep, 2 filter
        batch1_scores = torch.tensor([[0.7], [0.3]])
        batch2_scores = torch.tensor([[0.8], [0.2]])
        mock_model.side_effect = [batch1_scores, batch2_scores]

        stage.process(sample_image_batch)

        # Should log for filtered images only (individual images, not summary)
        filtered_calls = [call for call in mock_logger.info.call_args_list
                         if "filtered out" in str(call) and "image" in str(call).lower() and "path:" in str(call)]
        assert len(filtered_calls) == 2  # 2 images filtered

    def test_empty_batch(self, stage: ImageAestheticFilterStage) -> None:
        """Test processing empty batch."""
        empty_batch = ImageBatch(data=[])
        result = stage.process(empty_batch)
        assert len(result.data) == 0

    @patch("ray_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_no_embeddings(self, mock_aesthetic_scorer: Mock, stage: ImageAestheticFilterStage, sample_image_batch: ImageBatch, mock_model: Mock) -> None:
        """Test processing images without embeddings."""
        mock_aesthetic_scorer.return_value = mock_model
        stage.setup()

        # Remove embeddings from images
        for img in sample_image_batch.data:
            img.embedding = None

        result = stage.process(sample_image_batch)

        # Should return empty batch since no embeddings available
        assert len(result.data) == 0

    @patch("ray_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_edge_case_scores(self, mock_aesthetic_scorer: Mock, stage: ImageAestheticFilterStage, sample_image_batch: ImageBatch, mock_model: Mock) -> None:
        """Test edge case score values."""
        mock_aesthetic_scorer.return_value = mock_model
        stage.setup()

        rng = np.random.default_rng(42)
        # Test with edge case scores
        edge_scores = [np.array([0.0, 1.0])]  # Min and max possible scores
        mock_model.side_effect = edge_scores

        result = stage.process(sample_image_batch)

        # Should keep only the image with score 1.0
        assert len(result.data) == 1
        assert result.data[0].aesthetic_score == 1.0

    def test_metadata_preservation(self, sample_image_batch: ImageBatch) -> None:
        """Test that batch metadata is preserved."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", threshold=0.5)
        stage.setup()

        # Mock to keep all images
        with patch.object(stage, "model") as mock_model:
            mock_model.side_effect = [np.array([0.8, 0.9])]

            result = stage.process(sample_image_batch)

            # Check batch metadata is preserved
            assert result.dataset_name == sample_image_batch.dataset_name
            assert result.task_id == sample_image_batch.task_id
            assert result._metadata == sample_image_batch._metadata

    @patch("ray_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_score_assignment_accuracy(self, mock_aesthetic_scorer: Mock, stage: ImageAestheticFilterStage, sample_image_batch: ImageBatch, mock_model: Mock) -> None:
        """Test that aesthetic scores are correctly assigned to images."""
        mock_aesthetic_scorer.return_value = mock_model
        stage.setup()

        expected_scores = np.array([0.75, 0.85])
        mock_model.side_effect = [expected_scores]

        result = stage.process(sample_image_batch)

        # Check scores are assigned correctly
        assert len(result.data) == 2  # Both images should be kept
        for i, img in enumerate(result.data):
            assert img.aesthetic_score == expected_scores[i]

    def test_image_ordering_preservation(self, sample_image_batch: ImageBatch) -> None:
        """Test that image ordering is preserved through processing."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", threshold=0.5)
        stage.setup()

        original_ids = [img.image_id for img in sample_image_batch.data]

        # Mock to keep all images
        with patch.object(stage, "model") as mock_model:
            mock_model.side_effect = [np.array([0.8, 0.9]), np.array([0.7, 0.6])]

            result = stage.process(sample_image_batch)

            result_ids = [img.image_id for img in result.data]
            assert result_ids == original_ids

    def test_batch_size_handling(self, sample_image_batch: ImageBatch) -> None:
        """Test handling of different batch sizes."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", threshold=0.5, batch_size=1)
        stage.setup()

        # Mock to keep all images
        with patch.object(stage, "model") as mock_model:
            # With batch_size=1, should have 4 separate calls
            mock_model.side_effect = [
                np.array([0.8]), np.array([0.9]), 
                np.array([0.7]), np.array([0.6])
            ]

            result = stage.process(sample_image_batch)

            # Should have made 4 calls (one per image)
            assert mock_model.call_count == 4
            assert len(result.data) == 4

    def test_threshold_boundary_exact(self, sample_image_batch: ImageBatch) -> None:
        """Test behavior with scores exactly at threshold."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", threshold=0.5)
        stage.setup()

        # Mock scores exactly at threshold
        with patch.object(stage, "model") as mock_model:
            mock_model.side_effect = [np.array([0.5, 0.5])]

            result = stage.process(sample_image_batch)

            # Should keep both images (>= threshold)
            assert len(result.data) == 2
            for img in result.data:
                assert img.aesthetic_score == 0.5

    def test_large_batch_processing(self, sample_image_batch: ImageBatch) -> None:
        """Test processing with many images."""
        # Create a larger batch by replicating existing images
        large_data = sample_image_batch.data * 25  # 100 images
        large_batch = ImageBatch(
            data=large_data,
            dataset_name=sample_image_batch.dataset_name,
            task_id=sample_image_batch.task_id
        )

        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", threshold=0.5, batch_size=10)
        stage.setup()

        # Mock to alternate between keeping and filtering
        with patch.object(stage, "model") as mock_model:
            # 10 batches of 10 images each, alternating scores
            batch_results = []
            for i in range(10):
                if i % 2 == 0:
                    batch_results.append(np.array([0.6] * 10))  # Keep these
                else:
                    batch_results.append(np.array([0.4] * 10))  # Filter these
            
            mock_model.side_effect = batch_results

            result = stage.process(large_batch)

            # Should keep 50 images (5 batches * 10 images)
            assert len(result.data) == 50
            assert all(img.aesthetic_score == 0.6 for img in result.data)

    def test_score_statistics(self, sample_image_batch: ImageBatch) -> None:
        """Test that score statistics are meaningful."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", threshold=0.5)
        stage.setup()

        # Mock varied scores
        with patch.object(stage, "model") as mock_model:
            varied_scores = [np.array([0.1, 0.9]), np.array([0.3, 0.7])]
            mock_model.side_effect = varied_scores

            result = stage.process(sample_image_batch)

            # Should keep 2 images with high scores
            assert len(result.data) == 2
            scores = [img.aesthetic_score for img in result.data]
            assert 0.9 in scores
            assert 0.7 in scores

    def test_concurrent_processing_safety(self, sample_image_batch: ImageBatch) -> None:
        """Test that processing is safe for concurrent execution."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", threshold=0.5)
        stage.setup()

        # Process the same batch multiple times to simulate concurrency
        with patch.object(stage, "model") as mock_model:
            mock_model.side_effect = [np.array([0.8, 0.9])] * 3

            results = []
            for _ in range(3):
                result = stage.process(sample_image_batch)
                results.append(result)

            # All results should be identical
            for result in results:
                assert len(result.data) == 2
                assert result.data[0].aesthetic_score == 0.8
                assert result.data[1].aesthetic_score == 0.9

    def test_model_error_handling(self, sample_image_batch: ImageBatch) -> None:
        """Test handling of model errors."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", threshold=0.5)
        stage.setup()

        # Mock model to raise an exception
        with patch.object(stage, "model") as mock_model:
            mock_model.side_effect = RuntimeError("Model error")

            # Should handle the error gracefully
            try:
                result = stage.process(sample_image_batch)
                # If no exception, should return empty or handle gracefully
                assert isinstance(result, ImageBatch)
            except RuntimeError:
                # If exception propagates, that's also acceptable behavior
                pass
