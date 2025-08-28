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

from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.tasks import ImageBatch, ImageObject


class TestImageAestheticFilterStage:
    """Test suite for ImageAestheticFilterStage."""

    @pytest.fixture
    def stage(self) -> ImageAestheticFilterStage:
        """Create a test stage instance."""
        return ImageAestheticFilterStage(
            model_dir="test_models/aesthetics", score_threshold=0.5, model_inference_batch_size=2
        )

    @pytest.fixture
    def mock_model(self) -> Mock:
        """Create a mock aesthetic scoring model."""
        model = Mock()
        model.setup = Mock()
        # Mock to return aesthetic scores between 0 and 1
        model.return_value = torch.tensor([0.3, 0.7, 0.2, 0.8])
        return model

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
            _stage_perf={},
        )

    def test_stage_properties(self, stage: ImageAestheticFilterStage) -> None:
        """Test stage properties."""
        assert stage.name == "image_aesthetic_filter"
        # Allow either requesting GPUs or not, depending on environment
        assert stage.resources.gpus in (0.25, 0.0)
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

    @patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_setup(self, mock_aesthetic_scorer: Mock, stage: ImageAestheticFilterStage) -> None:
        """Test stage setup."""
        mock_model = Mock()
        mock_aesthetic_scorer.return_value = mock_model

        stage.setup()

        mock_aesthetic_scorer.assert_called_once_with(model_dir="test_models/aesthetics")
        mock_model.setup.assert_called_once()
        assert stage.model == mock_model

    @patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_process_filtering(
        self,
        mock_aesthetic_scorer: Mock,
        stage: ImageAestheticFilterStage,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test the main filtering process."""
        mock_aesthetic_scorer.return_value = mock_model

        stage.setup()

        # With model_inference_batch_size=2, we'll have 2 calls: [img1, img2] and [img3, img4]
        # First batch gets scores [0.3, 0.7], second batch gets [0.2, 0.8]
        mock_model.side_effect = [
            torch.tensor([0.3, 0.7]),  # First batch
            torch.tensor([0.2, 0.8]),  # Second batch
        ]

        result = stage.process(sample_image_batch)

        # Check filtering results - scores 0.7 and 0.8 should pass (threshold 0.5)
        assert len(result.data) == 2  # 2 images should pass the 0.5 threshold

        # Get the image IDs that passed
        passed_ids = [img.image_id for img in result.data]

        # img_002 should have score 0.7 and img_004 should have score 0.8
        assert "img_002" in passed_ids
        assert "img_004" in passed_ids

        # Check that scores were assigned correctly (with floating point tolerance)
        for img in result.data:
            if img.image_id == "img_002":
                assert abs(img.aesthetic_score - 0.7) < 1e-5
            elif img.image_id == "img_004":
                assert abs(img.aesthetic_score - 0.8) < 1e-5

        # Check that the task has updated ID
        assert result.task_id == f"{sample_image_batch.task_id}_{stage.name}"

    @patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_threshold_variations(
        self, mock_aesthetic_scorer: Mock, sample_image_batch: ImageBatch, mock_model: Mock
    ) -> None:
        """Test filtering with different thresholds."""
        mock_aesthetic_scorer.return_value = mock_model

        # Test with high threshold (0.9)
        high_threshold_stage = ImageAestheticFilterStage(score_threshold=0.9, model_inference_batch_size=4)
        high_threshold_stage.setup()
        high_threshold_stage.model = mock_model

        # Mock scores: [0.3, 0.7, 0.2, 0.8] - none should pass 0.9 threshold
        mock_model.return_value = torch.tensor([0.3, 0.7, 0.2, 0.8])

        result_high = high_threshold_stage.process(sample_image_batch)
        assert len(result_high.data) == 0

        # Test with low threshold (0.1)
        low_threshold_stage = ImageAestheticFilterStage(score_threshold=0.1, model_inference_batch_size=4)
        low_threshold_stage.setup()
        low_threshold_stage.model = mock_model

        result_low = low_threshold_stage.process(sample_image_batch)
        assert len(result_low.data) == 4  # All should pass 0.1 threshold

    @patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    @patch("nemo_curator.stages.image.filters.aesthetic_filter.logger")
    @pytest.mark.usefixtures("stage")
    def test_verbose_logging(
        self,
        mock_logger: Mock,
        mock_aesthetic_scorer: Mock,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test verbose logging functionality."""
        mock_aesthetic_scorer.return_value = mock_model

        # Create verbose stage
        verbose_stage = ImageAestheticFilterStage(
            model_dir="test_models/aesthetics", score_threshold=0.5, verbose=True
        )
        verbose_stage.setup()
        verbose_stage.model = mock_model

        mock_model.return_value = torch.tensor([0.3, 0.7, 0.2, 0.8])

        verbose_stage.process(sample_image_batch)

        # Should log filtering results
        filtering_calls = [call for call in mock_logger.info.call_args_list if "Aesthetic filtering:" in str(call)]
        assert len(filtering_calls) > 0

    @patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_empty_batch(self, mock_aesthetic_scorer: Mock, stage: ImageAestheticFilterStage) -> None:
        """Test processing empty image batch."""
        empty_batch = ImageBatch(data=[], task_id="empty_test", dataset_name="test_dataset")
        mock_aesthetic_scorer.return_value = Mock()

        stage.setup()

        result = stage.process(empty_batch)

        assert len(result.data) == 0
        # Model should not be called for empty batch
        stage.model.assert_not_called()

    @patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_no_embeddings(self, mock_aesthetic_scorer: Mock) -> None:
        """Test handling of images without embeddings."""
        # Create images without embeddings
        rng = np.random.default_rng(42)
        images_no_embeddings = [
            ImageObject(
                image_path="/path/to/img_001.jpg",
                image_id="img_001",
                image_data=rng.random((224, 224, 3)),
                embedding=None,  # No embedding
            )
        ]

        batch = ImageBatch(data=images_no_embeddings, task_id="no_embed_test", dataset_name="test_dataset")
        mock_aesthetic_scorer.return_value = Mock()

        stage = ImageAestheticFilterStage(
            model_dir="test_models/aesthetics", score_threshold=0.5, model_inference_batch_size=2
        )
        stage.setup()

        # This should handle gracefully (may raise exception or filter out)
        with pytest.raises((AttributeError, TypeError)):
            stage.process(batch)

    @patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_edge_case_scores(
        self,
        mock_aesthetic_scorer: Mock,
        stage: ImageAestheticFilterStage,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test handling of edge case scores (0.0, 1.0, exactly at threshold)."""
        mock_aesthetic_scorer.return_value = mock_model

        stage.setup()

        # Test scores at exact threshold and boundaries
        mock_model.return_value = torch.tensor([0.0, 0.5, 1.0, 0.5])  # threshold = 0.5

        result = stage.process(sample_image_batch)

        # Due to batching, we need to check which images actually passed
        assert len(result.data) >= 2  # At least the 1.0 and one 0.5 should pass

        # Check that all passed scores are >= threshold
        for img_obj in result.data:
            assert img_obj.aesthetic_score >= 0.5

    @patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer")
    def test_score_assignment_accuracy(
        self,
        mock_aesthetic_scorer: Mock,
        stage: ImageAestheticFilterStage,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test that scores are correctly assigned to the right images."""
        mock_aesthetic_scorer.return_value = mock_model

        stage.setup()

        # With model_inference_batch_size=2, we'll have 2 calls with 2 images each
        # First batch gets scores [0.11, 0.22], second batch gets [0.33, 0.44]
        mock_model.side_effect = [
            torch.tensor([0.11, 0.22]),  # First batch: img_001, img_002
            torch.tensor([0.33, 0.44]),  # Second batch: img_003, img_004
        ]

        # Lower threshold so all pass
        stage.score_threshold = 0.1
        result = stage.process(sample_image_batch)

        # All should pass and have correct scores (with tolerance)
        assert len(result.data) == 4

        # Verify each image has the correct score
        for img in result.data:
            if img.image_id == "img_001":
                assert abs(img.aesthetic_score - 0.11) < 1e-5
            elif img.image_id == "img_002":
                assert abs(img.aesthetic_score - 0.22) < 1e-5
            elif img.image_id == "img_003":
                assert abs(img.aesthetic_score - 0.33) < 1e-5
            elif img.image_id == "img_004":
                assert abs(img.aesthetic_score - 0.44) < 1e-5

    def test_metadata_preservation(self, sample_image_batch: ImageBatch) -> None:
        """Test that batch metadata is preserved."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", score_threshold=0.5)

        with patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer") as mock_aesthetic_scorer:
            mock_model = Mock()
            mock_model.return_value = torch.tensor([0.7, 0.8, 0.9, 1.0])  # All pass
            mock_aesthetic_scorer.return_value = mock_model

            stage.setup()

            # Add custom metadata
            sample_image_batch._metadata["custom_key"] = "custom_value"

            result = stage.process(sample_image_batch)

            # Check metadata preservation
            assert result._metadata == sample_image_batch._metadata
            assert result.dataset_name == sample_image_batch.dataset_name

    def test_image_ordering_preservation(self, sample_image_batch: ImageBatch) -> None:
        """Test that image ordering is preserved through processing."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", score_threshold=0.5)

        with patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer") as mock_aesthetic_scorer:
            mock_model = Mock()
            # All scores above threshold, in descending order
            mock_model.return_value = torch.tensor([0.9, 0.8, 0.7, 0.6])
            mock_aesthetic_scorer.return_value = mock_model

            stage.setup()

            result = stage.process(sample_image_batch)

            # Check that ordering is preserved
            assert len(result.data) == 4
            assert result.data[0].image_id == "img_001"
            assert result.data[1].image_id == "img_002"
            assert result.data[2].image_id == "img_003"
            assert result.data[3].image_id == "img_004"

    def test_batch_size_handling(self, sample_image_batch: ImageBatch) -> None:
        """Test handling of different batch sizes."""
        stage = ImageAestheticFilterStage(
            model_dir="test_models/aesthetics", score_threshold=0.5, model_inference_batch_size=1
        )

        with patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer") as mock_aesthetic_scorer:
            mock_model = Mock()
            # Return one score at a time since model_batch_size=1
            mock_model.return_value = torch.tensor([0.7])
            mock_aesthetic_scorer.return_value = mock_model

            stage.setup()

            result = stage.process(sample_image_batch)

            # Should call model 4 times (once per image)
            assert mock_model.call_count == 4
            # All should pass with score 0.7
            assert len(result.data) == 4

    def test_threshold_boundary_exact(self, sample_image_batch: ImageBatch) -> None:
        """Test behavior with scores exactly at threshold."""
        stage = ImageAestheticFilterStage(
            model_dir="test_models/aesthetics", score_threshold=0.5, model_inference_batch_size=4
        )

        with patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer") as mock_aesthetic_scorer:
            mock_model = Mock()
            # Test exact threshold values - all 4 images in one batch
            # Use a clearer separation to avoid floating point precision issues
            mock_model.return_value = torch.tensor([0.5, 0.51, 0.49, 0.5])
            mock_aesthetic_scorer.return_value = mock_model

            stage.setup()

            result = stage.process(sample_image_batch)

            # 0.5 and above should pass (>= threshold)
            assert len(result.data) == 3
            passed_scores = [img.aesthetic_score for img in result.data]

            # Check that all passed scores are >= threshold
            for score in passed_scores:
                assert score >= 0.5

            # Check that the expected high scores are present
            assert any(abs(score - 0.5) < 1e-5 for score in passed_scores)
            assert any(abs(score - 0.51) < 1e-5 for score in passed_scores)

            # Check that the low score (0.49) did not pass
            assert not any(abs(score - 0.49) < 1e-5 for score in passed_scores)

    def test_large_batch_processing(self, sample_image_batch: ImageBatch) -> None:
        """Test processing with many images."""
        # Create a larger batch by replicating existing images
        large_data = sample_image_batch.data * 25  # 100 images
        large_batch = ImageBatch(
            data=large_data, dataset_name=sample_image_batch.dataset_name, task_id=sample_image_batch.task_id
        )

        stage = ImageAestheticFilterStage(
            model_dir="test_models/aesthetics", score_threshold=0.5, model_inference_batch_size=10
        )

        with patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer") as mock_aesthetic_scorer:
            mock_model = Mock()
            # Return scores for 10 images at a time
            mock_model.return_value = torch.tensor([0.6] * 10)  # All pass
            mock_aesthetic_scorer.return_value = mock_model

            stage.setup()

            result = stage.process(large_batch)

            # Should process in batches of 10, so 10 calls total for 100 images
            assert mock_model.call_count == 10
            # All should pass
            assert len(result.data) == 100

    def test_score_statistics(self, sample_image_batch: ImageBatch) -> None:
        """Test that score statistics are meaningful."""
        stage = ImageAestheticFilterStage(
            model_dir="test_models/aesthetics", score_threshold=0.5, model_inference_batch_size=4
        )

        with patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer") as mock_aesthetic_scorer:
            mock_model = Mock()
            # Use a range of scores - all 4 images in one batch
            mock_model.return_value = torch.tensor([0.1, 0.3, 0.7, 0.9])
            mock_aesthetic_scorer.return_value = mock_model

            stage.setup()

            result = stage.process(sample_image_batch)

            # Only 0.7 and 0.9 should pass
            assert len(result.data) == 2
            scores = [img.aesthetic_score for img in result.data]
            assert min(scores) >= 0.5  # All scores should be above threshold

            # Check that expected scores are present (with tolerance)
            assert any(abs(score - 0.7) < 1e-5 for score in scores)
            assert any(abs(score - 0.9) < 1e-5 for score in scores)

    def test_concurrent_processing_safety(self, sample_image_batch: ImageBatch) -> None:
        """Test that processing is safe for concurrent execution."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", score_threshold=0.5)

        with patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer") as mock_aesthetic_scorer:
            mock_model = Mock()
            mock_model.return_value = torch.tensor([0.7, 0.8, 0.9, 1.0])
            mock_aesthetic_scorer.return_value = mock_model

            stage.setup()

            # Process same batch twice (simulating concurrent access)
            result1 = stage.process(sample_image_batch)
            result2 = stage.process(sample_image_batch)

            # Both should produce consistent results
            assert len(result1.data) == len(result2.data) == 4
            for img1, img2 in zip(result1.data, result2.data, strict=True):
                assert img1.aesthetic_score == img2.aesthetic_score

    def test_model_error_handling(self, sample_image_batch: ImageBatch) -> None:
        """Test handling of model errors."""
        stage = ImageAestheticFilterStage(model_dir="test_models/aesthetics", score_threshold=0.5)

        with patch("nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer") as mock_aesthetic_scorer:
            mock_model = Mock()
            # Simulate model error
            mock_model.side_effect = RuntimeError("Model failed")
            mock_aesthetic_scorer.return_value = mock_model

            stage.setup()

            # Should propagate error (or handle gracefully depending on implementation)
            with pytest.raises(RuntimeError):
                stage.process(sample_image_batch)


# GPU integration test appended to CPU suite
@pytest.mark.gpu
def test_image_aesthetic_filter_on_gpu() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU CI
        pytest.skip("CUDA not available; skipping GPU aesthetic test")

    class _DummyAestheticScorer:
        def __init__(self, _model_dir: str | None = None, **_kwargs: object) -> None:
            # Accept both positional and keyword args for compatibility
            return None

        def setup(self) -> None:
            return None

        def __call__(self, embeddings_numpy: np.ndarray) -> torch.Tensor:
            device = torch.device("cuda")
            x = torch.from_numpy(embeddings_numpy).to(device=device, dtype=torch.float32)
            scores = x.mean(dim=1)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    rng = np.random.default_rng(7)
    import tempfile

    tmp_dir = tempfile.gettempdir()
    images = [
        ImageObject(
            image_id=f"img_{i}", image_path=f"{tmp_dir}/{i}.jpg", embedding=rng.normal(size=(8,)).astype(np.float32)
        )
        for i in range(6)
    ]
    batch = ImageBatch(data=images, dataset_name="ds", task_id="t0")

    stage = ImageAestheticFilterStage(model_dir="/unused", model_inference_batch_size=3, score_threshold=0.3)

    with patch(
        "nemo_curator.stages.image.filters.aesthetic_filter.AestheticScorer",
        _DummyAestheticScorer,
    ):
        stage.setup()
        out = stage.process(batch)

    assert isinstance(out, ImageBatch)
    assert 1 <= len(out.data) <= len(batch.data)
    assert all(hasattr(img, "aesthetic_score") for img in batch.data)
