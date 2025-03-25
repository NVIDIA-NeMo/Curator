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

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from nemo_curator.utils.import_utils import gpu_only_import_from

# These imports should only work on GPU systems
ImageTextPairDataset = gpu_only_import_from(
    "nemo_curator.datasets.image_text_pair_dataset", "ImageTextPairDataset"
)
TimmImageEmbedder = gpu_only_import_from(
    "nemo_curator.image.embedders.timm", "TimmImageEmbedder"
)


# Test initialization parameters
@pytest.mark.gpu
def test_init_defaults():
    """Test that TimmImageEmbedder initializes with default parameters correctly."""
    embedder = TimmImageEmbedder(model_name="resnet18")
    assert embedder.model_name == "resnet18"
    assert embedder.pretrained is False
    assert embedder.batch_size == 1
    assert embedder.num_threads_per_worker == 4
    assert embedder.image_embedding_column == "image_embedding"
    assert embedder.normalize_embeddings is True
    assert embedder.autocast is True
    assert embedder.use_index_files is False
    assert len(embedder.classifiers) == 0


@pytest.mark.gpu
def test_init_custom_params():
    """Test that TimmImageEmbedder initializes with custom parameters correctly."""
    embedder = TimmImageEmbedder(
        model_name="vit_base_patch16_224",
        pretrained=True,
        batch_size=64,
        num_threads_per_worker=8,
        image_embedding_column="custom_embedding",
        normalize_embeddings=False,
        autocast=False,
        use_index_files=True,
    )
    assert embedder.model_name == "vit_base_patch16_224"
    assert embedder.pretrained is True
    assert embedder.batch_size == 64
    assert embedder.num_threads_per_worker == 8
    assert embedder.image_embedding_column == "custom_embedding"
    assert embedder.normalize_embeddings is False
    assert embedder.autocast is False
    assert embedder.use_index_files is True


# Test _configure_forward method
@pytest.mark.gpu
def test_configure_forward(gpu_client):
    """Test that the _configure_forward method appropriately wraps the model's forward method."""
    embedder = TimmImageEmbedder(model_name="resnet18")

    # Create a mock model with a forward method
    class MockModel:
        def __init__(self):
            self.forward_called = False
            self.forward_args = None
            self.forward_kwargs = None

        def forward(self, *args, **kwargs):
            self.forward_called = True
            self.forward_args = args
            self.forward_kwargs = kwargs
            # Return a tensor to simulate model output
            return torch.ones((2, 512), device="cuda")

    mock_model = MockModel()

    # Configure the forward method
    configured_model = embedder._configure_forward(mock_model)

    # Call the configured forward method
    test_input = torch.ones((2, 3, 224, 224), device="cuda")
    output = configured_model.forward(test_input)

    # Check that the original forward was called
    assert mock_model.forward_called

    # Check that output is normalized (should have unit norm)
    if embedder.normalize_embeddings:
        # Calculate L2 norm of each row
        norms = torch.norm(output, p=2, dim=1)
        # All norms should be very close to 1.0
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    # Check output dtype (should be float32 regardless of input)
    assert output.dtype == torch.float32


# Test load_embedding_model method
@pytest.mark.gpu
def test_load_embedding_model(gpu_client):
    """Test that the load_embedding_model method correctly loads a timm model."""
    # Use a small model for this test
    embedder = TimmImageEmbedder(model_name="resnet18", pretrained=False)

    # Mock timm.create_model to avoid actual model loading
    with mock.patch("timm.create_model") as mock_create_model:
        # Create a simple mock model
        class MockTimmModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.eval_called = False
                self.to_device = None

            def eval(self):
                self.eval_called = True
                return self

            def to(self, device):
                self.to_device = device
                return self

            def forward(self, x):
                return torch.ones((x.shape[0], 512), device=x.device)

        mock_model = MockTimmModel()
        mock_create_model.return_value = mock_model

        # Call the method
        model = embedder.load_embedding_model(device="cuda")

        # Check that timm.create_model was called with correct args
        mock_create_model.assert_called_once_with("resnet18", pretrained=False)

        # Check that model.eval() was called
        assert mock_model.eval_called

        # Check that model.to() was called with the right device
        assert mock_model.to_device == "cuda"


# Test load_dataset_shard method with mocked DALI
@pytest.mark.gpu
def test_load_dataset_shard(gpu_client):
    """Test that load_dataset_shard correctly loads and processes dataset shards."""
    embedder = TimmImageEmbedder(model_name="resnet18", batch_size=1)

    # Use the real sample dataset
    sample_tar_path = Path(__file__).parent.parent.parent / "image_data" / "00000.tar"

    # Ensure the file exists
    assert sample_tar_path.exists(), f"Sample dataset not found at {sample_tar_path}"

    # Load the dataset
    generator = embedder.load_dataset_shard(str(sample_tar_path))

    # Get the batch
    images, metadata = next(generator)

    # Verify batch shapes and metadata
    assert images.shape[0] == 1  # Batch size of 1
    assert len(metadata) == 1
    assert metadata[0]["id"] == "0"
    assert "caption" in metadata[0]

    # Verify that we've exhausted the iterator (only one sample)
    with pytest.raises(StopIteration):
        next(generator)


# Test complete workflow with mocks (this tests the __call__ method indirectly)
@pytest.mark.gpu
def test_embedder_workflow(gpu_client):
    """Test the complete workflow of the TimmImageEmbedder."""
    # Get the real sample dataset path
    sample_tar_path = Path(__file__).parent.parent.parent / "image_data" / "00000.tar"

    # Ensure the file exists
    assert sample_tar_path.exists(), f"Sample dataset not found at {sample_tar_path}"

    # Setup a minimal ImageTextPairDataset with just one tar file
    data_dir = str(sample_tar_path.parent)

    # Mock metadata for our dataset (would normally be read from parquet files)
    metadata_dict = {
        "id": ["0"],
        "caption": [
            "A wine bottle outfitted with two forks in its cork and a duck head on top."
        ],
    }

    # Create mock metadata DataFrame
    with (
        mock.patch("dask_cudf.read_parquet"),
        mock.patch.object(
            ImageTextPairDataset, "_get_tar_files", return_value=[str(sample_tar_path)]
        ),
        mock.patch.object(
            ImageTextPairDataset, "_sort_partition", side_effect=lambda df, id_col: df
        ),
    ):
        # Create a mock metadata object
        mock_metadata = mock.MagicMock()
        mock_metadata.dtypes.to_dict.return_value = {
            "id": "object",
            "caption": "object",
        }
        mock_metadata.map_partitions.return_value = mock.MagicMock()

        # Create a test dataset
        test_dataset = mock.MagicMock(spec=ImageTextPairDataset)
        test_dataset.path = data_dir
        test_dataset.id_col = "id"
        test_dataset.tar_files = [str(sample_tar_path)]
        test_dataset.metadata = mock_metadata

        # Create the embedder
        embedder = TimmImageEmbedder(
            model_name="resnet18", batch_size=1, image_embedding_column="embeddings"
        )

        with (
            mock.patch(
                "nemo_curator.image.embedders.base.ImageTextPairDataset"
            ) as mock_dataset_class,
        ):
            # Call the embedder
            result = embedder(test_dataset)

            # Check that map_partitions was called
            mock_metadata.map_partitions.assert_called_once()

            # Verify the right metadata was passed
            meta_arg = mock_metadata.map_partitions.call_args[1]["meta"]
            assert "embeddings" in meta_arg

            # Verify that a new dataset was created and returned
            mock_dataset_class.assert_called_once()


# Test with non-default configurations
@pytest.mark.gpu
def test_with_disabled_normalization(gpu_client):
    """Test that embeddings aren't normalized when normalize_embeddings=False."""
    embedder = TimmImageEmbedder(model_name="resnet18", normalize_embeddings=False)

    # Create a mock model that returns a non-normalized tensor
    class MockModel:
        def forward(self, *args, **kwargs):
            # Create a tensor with non-unit norm
            return torch.ones((2, 512), device="cuda") * 2.0

    mock_model = MockModel()

    # Configure the forward method
    configured_model = embedder._configure_forward(mock_model)

    # Call the configured forward method
    test_input = torch.ones((2, 3, 224, 224), device="cuda")
    output = configured_model.forward(test_input)

    # Check that output is NOT normalized (should NOT have unit norm)
    norms = torch.norm(output, p=2, dim=1)
    # All norms should be close to 2.0 * sqrt(512)
    expected_norm = 2.0 * (512**0.5)
    assert torch.allclose(norms, torch.ones_like(norms) * expected_norm, atol=1e-6)


@pytest.mark.gpu
def test_with_disabled_autocast(gpu_client):
    """Test behavior when autocast is disabled."""
    embedder = TimmImageEmbedder(model_name="resnet18", autocast=False)

    # Create a mock model to verify autocast isn't being used
    class MockModel:
        def __init__(self):
            self.forward_called = False

        def forward(self, *args, **kwargs):
            self.forward_called = True
            return torch.ones((2, 512), device="cuda")

    mock_model = MockModel()

    # Mock torch.autocast to verify it's not called
    with mock.patch("torch.autocast") as mock_autocast:
        # Configure the forward method
        configured_model = embedder._configure_forward(mock_model)

        # Call the configured forward method
        test_input = torch.ones((2, 3, 224, 224), device="cuda")
        output = configured_model.forward(test_input)

        # Verify that autocast wasn't called
        mock_autocast.assert_not_called()

        # Verify that the model's forward was still called
        assert mock_model.forward_called


@pytest.mark.gpu
def test_with_index_files(gpu_client):
    """Test that index files are correctly used when enabled."""
    embedder = TimmImageEmbedder(model_name="resnet18", use_index_files=True)

    # Use the real sample dataset and index file
    sample_tar_path = Path(__file__).parent.parent.parent / "image_data" / "00000.tar"
    sample_idx_path = Path(__file__).parent.parent.parent / "image_data" / "00000.idx"

    # Ensure the files exist
    assert sample_tar_path.exists(), f"Sample dataset not found at {sample_tar_path}"
    assert sample_idx_path.exists(), f"Sample index file not found at {sample_idx_path}"

    # Load the dataset
    generator = embedder.load_dataset_shard(str(sample_tar_path))

    # Get the batch
    images, metadata = next(generator)

    # Verify batch shapes and metadata
    assert images.shape[0] == 1  # Batch size of 1
    assert len(metadata) == 1
    assert metadata[0]["id"] == "0"
    assert "caption" in metadata[0]

    # Verify that we've exhausted the iterator (only one sample)
    with pytest.raises(StopIteration):
        next(generator)

    # Verify the embedder has the flag set
    assert embedder.use_index_files


@pytest.mark.gpu
def test_run_inference_with_mock_model(gpu_client):
    """Test the _run_inference method directly with a mock model and mock dataset."""
    import cudf

    # Create a mock model that returns predictable embeddings
    class MockModel:
        def __call__(self, batch):
            # Return a tensor with predictable values based on batch size
            batch_size = batch.shape[0]
            embeddings = torch.ones((batch_size, 128), device="cuda") * 0.5
            # Normalize the embeddings directly in the __call__ method
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            return embeddings

    # Create a mock classifier
    class MockClassifier:
        def __init__(self):
            self.model_name = "mock_classifier"
            self.pred_column = "mock_classifier_scores"
            self.pred_type = "float32"

        def load_model(self, device):
            return lambda x: torch.ones((x.shape[0], 1), device="cuda") * 0.75

        def postprocess(self, series):
            return series

    # Create a mock TimmImageEmbedder that overrides the methods we need to control
    class MockTimmImageEmbedder(TimmImageEmbedder):
        def __init__(self, **kwargs):
            # Avoid calling the parent's __init__ which tries to load a real model
            # Just set the attributes we need directly
            self.model_name = kwargs.get("model_name", "resnet18")
            self.pretrained = kwargs.get("pretrained", False)
            self.batch_size = kwargs.get("batch_size", 1)
            self.num_threads_per_worker = kwargs.get("num_threads_per_worker", 4)
            self.image_embedding_column = kwargs.get(
                "image_embedding_column", "image_embedding"
            )
            self.normalize_embeddings = kwargs.get("normalize_embeddings", True)
            self.autocast = kwargs.get("autocast", True)
            self.use_index_files = kwargs.get("use_index_files", False)
            self.classifiers = kwargs.get("classifiers", [])
            self.mock_model = MockModel()
            self.mock_dataset_yielded = False
            # Skip the call to timm.create_model and the transformation setup

        def load_embedding_model(self, device):
            return self.mock_model

        def load_dataset_shard(self, tar_path):
            # Yield only once to avoid infinite loop
            if not self.mock_dataset_yielded:
                self.mock_dataset_yielded = True
                # Create a small batch of fake images and metadata
                batch = torch.ones((2, 3, 224, 224), device="cuda")
                metadata = [{"id": "1"}, {"id": "0"}]  # Deliberately out of order
                yield batch, metadata

        def _configure_forward(self, model):
            # Use a simplified version without trying to access the original model's forward
            def custom_forward(*args, **kwargs):
                # Just call the model directly
                image_features = model(*args, **kwargs)

                if self.normalize_embeddings:
                    image_features = torch.nn.functional.normalize(
                        image_features, dim=-1
                    )

                return image_features.to(torch.float32)

            # Replace the actual forward method with our custom one
            original_model = model
            original_model.forward = custom_forward
            return original_model

    # Create mock data
    partition = cudf.DataFrame(
        {"id": ["0", "1"], "caption": ["test caption 1", "test caption 2"]}
    )
    tar_paths = ["mock_tar_path.tar"]
    id_col = "id"
    partition_info = {"number": 0}

    # Mock the load_object_on_worker function to return our model directly
    with mock.patch(
        "nemo_curator.image.embedders.base.load_object_on_worker"
    ) as mock_load:
        # Configure the mock to return the model or classifier when called
        mock_load.side_effect = lambda name, fn, args: (
            MockModel()
            if name == "mock_model"
            else (lambda x: torch.ones((x.shape[0], 1), device="cuda") * 0.75)
        )

        # Test without classifiers
        embedder = MockTimmImageEmbedder(
            model_name="mock_model", image_embedding_column="embeddings", batch_size=2
        )

        # Call _run_inference directly
        result_partition = embedder._run_inference(
            partition, tar_paths, id_col, partition_info
        )

        # Verify that embeddings were added and ordered correctly
        assert "embeddings" in result_partition.columns

        # The embeddings are stored as lists in the cuDF DataFrame
        # We need to handle them differently than with to_numpy()
        embeddings_series = result_partition["embeddings"]
        assert len(embeddings_series) == 2

        # Test characteristics of each embedding by accessing individual items
        for i in range(len(embeddings_series)):
            # Extract the embedding as a list (or check if it needs conversion first)
            emb = embeddings_series.iloc[i]

            # Verify the embedding has the expected properties
            assert len(emb) == 128

            # Convert to numpy array for easier testing
            emb_array = np.array(emb)

            # The embeddings should have consistent values and be normalized
            # First check that all values are the same (since our input was uniform)
            assert np.allclose(emb_array, emb_array[0], atol=1e-6)

            # Check that the vector is normalized (unit norm)
            assert np.isclose(np.linalg.norm(emb_array), 1.0, atol=1e-6)

            # For a normalized vector of 128 identical values, each value should be 1/sqrt(128)
            expected_normalized_value = 1.0 / np.sqrt(128)
            assert np.isclose(emb_array[0], expected_normalized_value, atol=1e-6)

        # Test with classifier
        embedder = MockTimmImageEmbedder(
            model_name="mock_model",
            image_embedding_column="embeddings",
            batch_size=2,
            classifiers=[MockClassifier()],
        )
        embedder.mock_dataset_yielded = False  # Reset for reuse

        # Call _run_inference directly
        result_partition = embedder._run_inference(
            partition, tar_paths, id_col, partition_info
        )

        # Verify that classifier scores were added
        assert "mock_classifier_scores" in result_partition.columns

        # Handle the classifier scores in the same way
        classifier_scores_series = result_partition["mock_classifier_scores"]
        assert len(classifier_scores_series) == 2

        # Test characteristics of each classifier score
        for i in range(len(classifier_scores_series)):
            scores = classifier_scores_series.iloc[i]
            assert len(scores) == 1

            # Convert to numpy array for testing
            scores_array = np.array(scores)
            assert np.allclose(scores_array, 0.75, atol=1e-6)
