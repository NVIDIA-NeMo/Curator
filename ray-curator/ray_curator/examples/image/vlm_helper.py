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

import io
import os
import pathlib
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

import numpy as np
import requests
from loguru import logger
from PIL import Image

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import ImageBatch, ImageObject, _EmptyTask
from ray_curator.utils.file_utils import get_all_files_paths_under


@dataclass
class MultiSourceImageReaderStage(ProcessingStage[_EmptyTask, ImageBatch]):
    """
    Stage that reads images from multiple sources: local files or URLs.
    """

    input_path: str
    source_type: Literal["local", "urls"] = "local"
    image_limit: int = -1
    batch_size: int = 100
    verbose: bool = True
    _name: str = "multi_source_image_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_data", "image_path", "image_id"]

    def _is_image_file(self, path: str) -> bool:
        """Check if file has image extension."""
        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
            ".tif",
            ".tiff",
            ".webp",
            ".ppm",
            ".pgm",
            ".pnm",
        }
        return pathlib.Path(path).suffix.lower() in image_extensions

    def _is_valid_url(self, url: str) -> bool:
        """Check if string is a valid URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except (ValueError, TypeError):
            return False

    def _load_image_from_path(self, image_path: str, image_id: str) -> ImageObject | None:
        """Load image from local file path."""
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                image_data = np.array(img_rgb, dtype=np.uint8)

            return ImageObject(image_path=image_path, image_id=image_id, image_data=image_data)
        except (OSError, ValueError) as e:
            if self.verbose:
                logger.error(f"Error loading image {image_path}: {e}")
            return None

    def _load_image_from_url(self, image_url: str, image_id: str) -> ImageObject | None:
        """Load image from URL."""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            img = Image.open(io.BytesIO(response.content))
            img_rgb = img.convert("RGB")
            image_data = np.array(img_rgb, dtype=np.uint8)

            return ImageObject(image_path=image_url, image_id=image_id, image_data=image_data)
        except (requests.RequestException, OSError, ValueError) as e:
            if self.verbose:
                logger.error(f"Error loading image from URL {image_url}: {e}")
            return None

    def _load_images_from_local(self) -> list[ImageObject]:
        """Load images from local directory or file."""
        images = []

        if os.path.isfile(self.input_path):
            # Single file
            if self._is_image_file(self.input_path):
                image_id = pathlib.Path(self.input_path).stem
                image_obj = self._load_image_from_path(self.input_path, image_id)
                if image_obj:
                    images.append(image_obj)
        else:
            # Directory
            image_paths = get_all_files_paths_under(
                self.input_path,
                recurse_subdirectories=True,
                keep_extensions=[
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".gif",
                    ".tif",
                    ".tiff",
                    ".webp",
                    ".ppm",
                    ".pgm",
                    ".pnm",
                ],
            )

            for i, image_path in enumerate(image_paths):
                if self.image_limit > 0 and i >= self.image_limit:
                    break

                image_id = pathlib.Path(image_path).stem
                image_obj = self._load_image_from_path(image_path, image_id)
                if image_obj:
                    images.append(image_obj)

        return images

    def _load_images_from_urls(self) -> list[ImageObject]:
        """Load images from URL list file."""
        images = []

        try:
            with open(self.input_path) as f:
                urls = [line.strip() for line in f if line.strip()]

            for i, url in enumerate(urls):
                if self.image_limit > 0 and i >= self.image_limit:
                    break

                if self._is_valid_url(url):
                    image_id = f"url_image_{i}"
                    image_obj = self._load_image_from_url(url, image_id)
                    if image_obj:
                        images.append(image_obj)
                elif self.verbose:
                    logger.warning(f"Invalid URL: {url}")

        except OSError as e:
            if self.verbose:
                logger.error(f"Error reading URL file {self.input_path}: {e}")

        return images

    def _create_image_batches(self, all_images: list[ImageObject]) -> list[ImageBatch]:
        """Create ImageBatch objects from list of ImageObjects."""
        image_batches = []

        for i in range(0, len(all_images), self.batch_size):
            batch_images = all_images[i : i + self.batch_size]

            image_batch = ImageBatch(
                task_id=f"image_batch_{i // self.batch_size}", dataset_name=self.input_path, data=batch_images
            )
            image_batches.append(image_batch)

        if self.verbose:
            logger.info(f"Created {len(image_batches)} ImageBatch objects")

        return image_batches

    def process(self, _: _EmptyTask) -> list[ImageBatch]:
        """Process images from the specified source."""
        if self.verbose:
            logger.info(
                "Loading images from %s: %s",
                self.source_type,
                self.input_path,
            )

        if self.source_type == "local":
            images = self._load_images_from_local()
        elif self.source_type == "urls":
            images = self._load_images_from_urls()
        elif self.source_type == "webdataset":
            # Explicitly disallow here; handled by ImageReaderStage in pipeline
            if self.verbose:
                logger.error(
                    "source_type 'webdataset' is not supported in "
                    "MultiSourceImageReaderStage; use ImageReaderStage in the pipeline"
                )
            return []
        else:
            if self.verbose:
                logger.error(f"Unknown source type: {self.source_type}")
            return []

        if self.verbose:
            logger.info(f"Loaded {len(images)} images")

        return self._create_image_batches(images)
