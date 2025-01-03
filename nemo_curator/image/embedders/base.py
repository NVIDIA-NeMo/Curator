# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import cupy as cp
import torch
from crossfit.backend.cudf.series import create_list_series_from_1d_or_2d_ar
from tqdm import tqdm

from nemo_curator.datasets import ImageTextPairDataset
from nemo_curator.image.classifiers import ImageClassifier
from nemo_curator.utils.distributed_utils import load_object_on_worker


class ImageEmbedder(ABC):
    """
    An abstract base class for generating image embeddings.

    Subclasses only need to define how a model is loaded and a dataset
    is read in from a tar file shard. This class handles distributing
    the tasks across workers and saving the metadata to the dataset.
    The embedding model must be able to fit onto a single GPU.
    """

    def __init__(
        self,
        model_name: str,
        image_embedding_column: str,
        classifiers: Iterable[ImageClassifier],
    ) -> None:
        """
        Constructs an image embedder.

        Args:
            model_name (str): A unqiue name to identify the model on each worker
                and in the logs.
            image_embedding_column (str): The column name to be added where the
                image embeddings will be saved.
            classifiers (Iterable[ImageClassifier]): A collection of classifiers. If
                the iterable has a nonzero length, all classifiers will be loaded
                on the GPU at the same time and be passed the image embeddings
                immediately after they are created.
        """
        self.model_name = model_name
        self.image_embedding_column = image_embedding_column
        self.classifiers = classifiers

    def __call__(self, dataset: ImageTextPairDataset) -> ImageTextPairDataset:
        """
        Generates image embeddings for all images in the dataset.

        Args:
            dataset (ImageTextPairDataset): The dataset to create image embeddings for.

        Returns:
            ImageTextPairDataset: A dataset with image embeddings and potentially
                classifier scores.
        """
        meta = dataset.metadata.dtypes.to_dict()
        meta[self.image_embedding_column] = "object"
        for classifier in self.classifiers:
            meta[classifier.pred_column] = classifier.pred_type

        embedding_df = dataset.metadata.map_partitions(
            self._run_inference, dataset.tar_files, dataset.id_col, meta=meta
        )

        return ImageTextPairDataset(
            dataset.path,
            metadata=embedding_df,
            tar_files=dataset.tar_files,
            id_col=dataset.id_col,
        )

    def _run_inference(self, partition, tar_paths, id_col, partition_info=None):
        tar_path = tar_paths[partition_info["number"]]
        device = "cuda"

        model = load_object_on_worker(
            self.model_name,
            self.load_embedding_model,
            {"device": device},
        )
        classifier_models = []
        for classifier in self.classifiers:
            loaded_classifier = load_object_on_worker(
                classifier.model_name, classifier.load_model, {"device": device}
            )
            classifier_models.append(loaded_classifier)

        dataset = self.load_dataset_shard(tar_path)
        final_image_embeddings = []
        image_ids = []
        classifier_results = [[] for _ in self.classifiers]
        samples_completed = 0
        progress_bar = tqdm(
            total=len(partition),
            desc=f"{tar_path} - Embedding creation with {self.model_name}",
        )
        with torch.no_grad():
            for batch, metadata in dataset:
                image_embeddings = model(batch)
                final_image_embeddings.append(image_embeddings)
                image_ids.extend(m[id_col] for m in metadata)

                for classifier_model, results in zip(
                    classifier_models, classifier_results
                ):
                    classifier_result = classifier_model(image_embeddings)
                    results.append(classifier_result)

                batch_size = len(image_embeddings)
                samples_completed += batch_size
                progress_bar.update(batch_size)
        progress_bar.close()

        if samples_completed != len(partition):
            raise RuntimeError(
                f"Mismatch in sample count for partition {partition_info['number']}. "
                f"{len(partition)} samples found in the metadata, but {samples_completed} found in {tar_path}."
            )

        # Order the output of the shard
        sorted_indices = sorted(range(len(image_ids)), key=lambda k: image_ids[k])
        sorted_embeddings = torch.cat(final_image_embeddings, dim=0)[sorted_indices]

        concat_embedding_output = cp.asarray(sorted_embeddings)
        partition[self.image_embedding_column] = create_list_series_from_1d_or_2d_ar(
            concat_embedding_output, index=partition.index
        )

        for classifier, results in zip(self.classifiers, classifier_results):
            sorted_results = torch.cat(results, dim=0)[sorted_indices]
            concat_output = cp.asarray(sorted_results)
            series = create_list_series_from_1d_or_2d_ar(
                concat_output, index=partition.index
            )
            partition[classifier.pred_column] = classifier.postprocess(series)

        return partition

    @abstractmethod
    def load_dataset_shard(self, tar_path: str) -> Iterable:
        """
        Loads images and metadata from a tarfile in the dataset.

        Args:
            tar_path (str): The path to a tar file shard in the input WebDataset.

        Returns:
            Iterable: An iterator over the dataset. Each iteration should produce
                a tuple of (image, metadata) pairs. The batch of images will be passed
                directly to the model created by ImageEmbedder.load_embedding_model.
                The metadata must be a list of dictionaries. Each element of the list
                must correspond to the image in the batch at the same position.
                Each dictionary must contain a field that is the same as
                id_field in the dataset. This ID field in the metadata will be used
                to match the image to the its record in the metadata (Parquet) files.
        """
        pass

    @abstractmethod
    def load_embedding_model(self, device: str) -> Callable:
        """
        Loads the model used to generate image embeddings.

        Args:
            device (str): A PyTorch device identifier that specifies what GPU
                to load the model on.

        Returns:
            Callable: A callable model, usually a torch.nn.Module.
                The input to this model will be the batches of images output
                by the ImageEmbedder.load_dataset_shard.
        """
        pass
