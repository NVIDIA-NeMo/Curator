import pathlib
import tarfile
from dataclasses import dataclass

import numpy as np
from loguru import logger
from PIL import Image

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import ImageBatch, ImageObject, _EmptyTask
from ray_curator.utils.file_utils import get_all_files_paths_under


@dataclass
class ImageReaderStage(ProcessingStage[_EmptyTask, ImageBatch]):
    """Stage that reads webdataset tar files and loads images into ImageBatch objects."""
    input_dataset_path: str
    image_limit: int = -1
    batch_size: int = 100
    verbose: bool = True
    _name: str = "image_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_data", "image_path", "image_id"]

    def _find_tar_files(self) -> list[pathlib.Path]:
        """Find all tar files in the dataset directory."""
        if self.input_dataset_path is None:
            msg = "input_dataset_path is not set"
            raise ValueError(msg)

        tar_files = get_all_files_paths_under(
            self.input_dataset_path,
            recurse_subdirectories=True,
            keep_extensions=[".tar"],
        )

        if self.verbose:
            logger.info(f"Found {len(tar_files)} tar files under {self.input_dataset_path}")

        if len(tar_files) == 0:
            if self.verbose:
                logger.warning(f"No tar files found in {self.input_dataset_path}")
            return []

        # Convert string paths to pathlib.Path objects
        return [pathlib.Path(tar_path) if isinstance(tar_path, str) else tar_path for tar_path in tar_files]

    def _load_image_from_member(self, member: tarfile.TarInfo, tar: tarfile.TarFile, tar_path: pathlib.Path) -> ImageObject | None:
        """Load a single image from a tar member."""
        try:
            # Extract image key from filename (remove .jpg extension)
            image_key = member.name.replace(".jpg", "")

            # Load image from tar
            image_file = tar.extractfile(member)
            if image_file is None:
                if self.verbose:
                    logger.warning(f"Could not extract image {member.name} from {tar_path}")
                return None

            # Load image with PIL and convert to RGB numpy array
            with Image.open(image_file) as img:
                img_rgb = img.convert("RGB")
                image_data = np.array(img_rgb, dtype=np.uint8)

            # Create ImageObject with loaded data
            return ImageObject(
                image_path=str(tar_path / member.name),  # Virtual path in tar
                image_id=image_key,
                image_data=image_data
            )

        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            if self.verbose:
                logger.error(f"Error loading image {member.name} from {tar_path}: {e}")
            return None

    def _process_tar_file(self, tar_path: pathlib.Path, total_images_loaded: int) -> tuple[list[ImageObject], int]:
        """Process a single tar file and return loaded images."""
        images = []

        if self.verbose:
            logger.info(f"Processing tar file: {tar_path}")

        try:
            with tarfile.open(tar_path, "r") as tar:
                # Get all image files (jpg) in the tar
                image_members = [m for m in tar.getmembers() if m.name.endswith(".jpg") and m.isfile()]

                for member in image_members:
                    # Check image limit
                    if self.image_limit > 0 and total_images_loaded >= self.image_limit:
                        break

                    image_obj = self._load_image_from_member(member, tar, tar_path)
                    if image_obj is not None:
                        images.append(image_obj)
                        total_images_loaded += 1

        except (tarfile.ReadError, OSError, IOError) as e:
            if self.verbose:
                logger.error(f"Error processing tar file {tar_path}: {e}")

        return images, total_images_loaded

    def _create_image_batches(self, all_image_objects: list[ImageObject]) -> list[ImageBatch]:
        """Create ImageBatch objects from a list of ImageObjects."""
        image_batches = []
        for i in range(0, len(all_image_objects), self.batch_size):
            batch_images = all_image_objects[i:i + self.batch_size]

            image_batch = ImageBatch(
                task_id=f"image_batch_{i // self.batch_size}",
                dataset_name=self.input_dataset_path,
                data=batch_images
            )
            image_batches.append(image_batch)

        if self.verbose:
            logger.info(f"Created {len(image_batches)} ImageBatch objects with batch_size={self.batch_size}")

        return image_batches

    def process(self, _: _EmptyTask) -> list[ImageBatch]:
        """Process webdataset tar files and create ImageBatch objects."""
        tar_files = self._find_tar_files()
        if not tar_files:
            return []

        # Load all images from tar files
        all_image_objects = []
        total_images_loaded = 0

        for tar_file_path in tar_files:
            images, total_images_loaded = self._process_tar_file(tar_file_path, total_images_loaded)
            all_image_objects.extend(images)

            # Break if we hit the image limit
            if self.image_limit > 0 and total_images_loaded >= self.image_limit:
                break

        if self.verbose:
            logger.info(f"Loaded {total_images_loaded} images total from {len(tar_files)} tar files")

        if self.image_limit > 0 and total_images_loaded > self.image_limit:
            all_image_objects = all_image_objects[:self.image_limit]
            if self.verbose:
                logger.info(f"Limiting to first {self.image_limit} images")

        return self._create_image_batches(all_image_objects)
