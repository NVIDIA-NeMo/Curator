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

from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
from nemo_curator.tasks import ImageBatch, ImageObject


class TestImageNSFWFilterStage:
    """Test suite for ImageNSFWFilterStage."""

    @pytest.fixture
    def stage(self) -> ImageNSFWFilterStage:
        """Create a test stage instance."""
        return ImageNSFWFilterStage(
            model_dir="test_models/nsfw",
            score_threshold=0.5,
            model_inference_batch_size=2
        )

    @pytest.fixture
    def mock_model(self) -> Mock:
        """Create a mock NSFW scoring model."""
        model = Mock()
        model.setup = Mock()
        # Mock to return NSFW scores between 0 and 1
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
            _stage_perf={}
        )

    def test_stage_properties(self, stage: ImageNSFWFilterStage) -> None:
        """Test stage properties."""
        assert stage.name == "image_nsfw_filter"
        # Allow either requesting GPUs or not, depending on environment
        assert stage.resources.gpus in (0.25, 0.0)
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

    @patch("nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer")
    def test_setup(self, mock_nsfw_scorer: Mock, stage: ImageNSFWFilterStage) -> None:
        """Test stage setup."""
        mock_model = Mock()
        mock_nsfw_scorer.return_value = mock_model

        stage.setup()

        mock_nsfw_scorer.assert_called_once_with(model_dir="test_models/nsfw")
        mock_model.setup.assert_called_once()
        assert stage.model == mock_model

    @patch("nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer")
    def test_process_filtering(
        self,
        mock_nsfw_scorer: Mock,
        stage: ImageNSFWFilterStage,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test the main filtering process."""
        mock_nsfw_scorer.return_value = mock_model

        stage.setup()

        # With model_inference_batch_size=2, we'll have 2 calls: [img1, img2] and [img3, img4]
        # First batch gets scores [0.3, 0.7], second batch gets [0.2, 0.8]
        # NSFW filter keeps images with scores < threshold (0.5)
        # So keep img1 (0.3), img3 (0.2), filter out img2 (0.7), img4 (0.8)
        mock_model.side_effect = [
            torch.tensor([0.3, 0.7]),  # First batch
            torch.tensor([0.2, 0.8])   # Second batch
        ]

        result = stage.process(sample_image_batch)

        # Check filtering results - scores 0.3 and 0.2 should pass (< threshold 0.5)
        assert len(result.data) == 2  # 2 images should pass the 0.5 threshold

        # Get the image IDs that passed
        passed_ids = [img.image_id for img in result.data]

        # img_001 should have score 0.3 and img_003 should have score 0.2
        assert "img_001" in passed_ids
        assert "img_003" in passed_ids

        # Check that scores were assigned correctly (with floating point tolerance)
        for img in result.data:
            if img.image_id == "img_001":
                assert abs(img.nsfw_score - 0.3) < 1e-5
            elif img.image_id == "img_003":
                assert abs(img.nsfw_score - 0.2) < 1e-5

        # Check that the task has updated ID
        assert result.task_id == f"{sample_image_batch.task_id}_{stage.name}"

    @patch("nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer")
    def test_process_high_nsfw_filtering(
        self,
        mock_nsfw_scorer: Mock,
        stage: ImageNSFWFilterStage,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test filtering with high NSFW scores (all should be filtered)."""
        mock_nsfw_scorer.return_value = mock_model

        stage.setup()

        # All images have high NSFW scores (above threshold)
        mock_model.side_effect = [
            torch.tensor([0.8, 0.9]),  # First batch
            torch.tensor([0.7, 0.6])   # Second batch
        ]

        result = stage.process(sample_image_batch)

        # All images should be filtered out
        assert len(result.data) == 0

    def test_different_thresholds(self, sample_image_batch: ImageBatch) -> None:
        """Test filtering with different thresholds."""
        with patch("nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer") as mock_nsfw_scorer:
            mock_model = Mock()
            # Fixed scores: [0.2, 0.4, 0.6, 0.8]
            mock_model.return_value = torch.tensor([0.2, 0.4, 0.6, 0.8])
            mock_nsfw_scorer.return_value = mock_model

            # Test with strict threshold (0.3) - only first image should pass
            strict_stage = ImageNSFWFilterStage(score_threshold=0.3, model_inference_batch_size=4)
            strict_stage.setup()
            strict_result = strict_stage.process(sample_image_batch)
            assert len(strict_result.data) == 1

            # Test with lenient threshold (0.9) - all images should pass
            lenient_stage = ImageNSFWFilterStage(score_threshold=0.9, model_inference_batch_size=4)
            lenient_stage.setup()
            lenient_stage.model = mock_model
            lenient_result = lenient_stage.process(sample_image_batch)
            assert len(lenient_result.data) == 4

    @patch("nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer")
    def test_threshold_boundary_cases(
        self,
        mock_nsfw_scorer: Mock,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test boundary cases at threshold."""
        mock_nsfw_scorer.return_value = mock_model

        stage = ImageNSFWFilterStage(
            model_dir="test_models/nsfw",
            score_threshold=0.5,
            model_inference_batch_size=2
        )
        stage.setup()

        # Test scores around threshold (0.5)
        mock_model.side_effect = [
            torch.tensor([0.5, 0.49]),   # First batch: exactly at and just below
            torch.tensor([0.51, 0.499])  # Second batch: just above and just below
        ]

        result = stage.process(sample_image_batch)

        # Only scores < 0.5 should pass
        assert len(result.data) == 2
        for img in result.data:
            assert img.nsfw_score < 0.5

    @patch("nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer")
    def test_all_images_filtered(
        self,
        mock_nsfw_scorer: Mock,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test when all images are filtered out."""
        mock_nsfw_scorer.return_value = mock_model

        stage = ImageNSFWFilterStage(
            model_dir="test_models/nsfw",
            score_threshold=0.5,
            model_inference_batch_size=2
        )
        stage.setup()

        # All high NSFW scores
        mock_model.side_effect = [
            torch.tensor([0.9, 0.8]),  # First batch
            torch.tensor([0.7, 0.6])   # Second batch
        ]

        result = stage.process(sample_image_batch)

        assert len(result.data) == 0
        assert result.dataset_name == sample_image_batch.dataset_name
        assert result.task_id == f"{sample_image_batch.task_id}_{stage.name}"

    @patch("nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer")
    def test_no_images_filtered(
        self,
        mock_nsfw_scorer: Mock,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test when no images are filtered out."""
        mock_nsfw_scorer.return_value = mock_model

        stage = ImageNSFWFilterStage(
            model_dir="test_models/nsfw",
            score_threshold=0.5,
            model_inference_batch_size=2
        )
        stage.setup()

        # All low NSFW scores
        mock_model.side_effect = [
            torch.tensor([0.1, 0.2]),  # First batch
            torch.tensor([0.3, 0.4])   # Second batch
        ]

        result = stage.process(sample_image_batch)

        assert len(result.data) == 4
        for img in result.data:
            assert img.nsfw_score < 0.5

    @patch("nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer")
    def test_batch_processing(
        self,
        mock_nsfw_scorer: Mock,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test batch processing with different batch sizes."""
        mock_nsfw_scorer.return_value = mock_model

        # Test with model_batch_size=1 (process one image at a time)
        single_stage = ImageNSFWFilterStage(model_inference_batch_size=1, score_threshold=0.5)
        single_stage.setup()
        single_stage.model = mock_model

        # Mock returns one score at a time
        mock_model.return_value = torch.tensor([0.3])

        result = single_stage.process(sample_image_batch)

        # Should call model 4 times (once per image)
        assert mock_model.call_count == 4
        # All should pass with score 0.3
        assert len(result.data) == 4

    @patch("nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer")
    @patch("nemo_curator.stages.image.filters.nsfw_filter.logger")
    def test_verbose_logging(
        self,
        mock_logger: Mock,
        mock_nsfw_scorer: Mock,
        sample_image_batch: ImageBatch,
        mock_model: Mock,
    ) -> None:
        """Test verbose logging functionality."""
        mock_nsfw_scorer.return_value = mock_model

        # Create verbose stage with correct batch size
        verbose_stage = ImageNSFWFilterStage(
            model_dir="test_models/nsfw",
            score_threshold=0.5,
            model_inference_batch_size=2,  # Match the mock data structure
            verbose=True
        )
        verbose_stage.setup()
        verbose_stage.model = mock_model

        mock_model.side_effect = [
            torch.tensor([0.3, 0.7]),  # First batch: one pass, one fail
            torch.tensor([0.2, 0.8])   # Second batch: one pass, one fail
        ]

        verbose_stage.process(sample_image_batch)

        # Should log filtering results
        filtering_calls = [call for call in mock_logger.info.call_args_list
                          if "NSFW" in str(call)]
        assert len(filtering_calls) > 0

    @patch("nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer")
    def test_empty_batch(self, mock_nsfw_scorer: Mock, stage: ImageNSFWFilterStage) -> None:
        """Test processing empty image batch."""
        empty_batch = ImageBatch(data=[], task_id="empty_test", dataset_name="test_dataset")
        mock_nsfw_scorer.return_value = Mock()

        stage.setup()

        result = stage.process(empty_batch)

        assert len(result.data) == 0
        # Model should not be called for empty batch
        stage.model.assert_not_called()


# GPU integration test appended to CPU suite
@pytest.mark.gpu
def test_image_nsfw_filter_on_gpu() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU CI
        pytest.skip("CUDA not available; skipping GPU nsfw test")

    class _DummyNSFWScorer:
        def __init__(self, _model_dir: str | None = None, **_kwargs: object) -> None:
            # Accept both positional and keyword args for compatibility
            return None

        @staticmethod
        def download_weights_on_node(_model_dir: str | None = None) -> None:
            return None

        def setup(self) -> None:
            return None

        def __call__(self, embeddings_numpy: np.ndarray) -> torch.Tensor:
            device = torch.device("cuda")
            x = torch.from_numpy(embeddings_numpy).to(device=device, dtype=torch.float32)
            return torch.sigmoid(x.mean(dim=1))

    rng = np.random.default_rng(9)
    import tempfile

    tmp_dir = tempfile.gettempdir()
    images = [
        ImageObject(image_id=f"img_{i}", image_path=f"{tmp_dir}/{i}.jpg", embedding=rng.normal(size=(8,)).astype(np.float32))
        for i in range(6)
    ]
    batch = ImageBatch(data=images, dataset_name="ds", task_id="t0")

    stage = ImageNSFWFilterStage(model_dir="/unused", model_inference_batch_size=3, score_threshold=0.5)

    with patch(
        "nemo_curator.stages.image.filters.nsfw_filter.NSFWScorer",
        _DummyNSFWScorer,
    ):
        stage.setup_on_node()
        stage.setup()
        out = stage.process(batch)

    assert isinstance(out, ImageBatch)
    assert 1 <= len(out.data) <= len(batch.data)
    assert all(hasattr(img, "nsfw_score") for img in batch.data)
