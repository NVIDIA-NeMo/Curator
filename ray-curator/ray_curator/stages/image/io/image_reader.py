import pathlib
import tarfile
from dataclasses import dataclass

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import FileGroupTask, ImageBatch, ImageObject


@dataclass
class ImageReaderStage(ProcessingStage[FileGroupTask, ImageBatch]):
    """Stage that reads webdataset tar files and loads images into ImageBatch objects."""
    task_batch_size: int = 100  # Number of images per ImageBatch object
    verbose: bool = True
    _name: str = "image_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_data", "image_path", "image_id"]

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

        except (OSError, Image.UnidentifiedImageError) as e:
            if self.verbose:
                logger.error(f"Error loading image {member.name} from {tar_path}: {e}")
            return None

    def _process_tar_file(self, tar_path: pathlib.Path) -> list[ImageObject]:
        """Process a single tar file and return loaded images."""
        images = []

        try:
            with tarfile.open(tar_path, "r") as tar:
                # Get all image files (jpg) in the tar
                image_members = [m for m in tar.getmembers() if m.name.endswith(".jpg") and m.isfile()]

                for member in image_members:
                    image_obj = self._load_image_from_member(member, tar, tar_path)
                    if image_obj is not None:
                        images.append(image_obj)

        except (tarfile.ReadError, OSError) as e:
            if self.verbose:
                logger.error(f"Error processing tar file {tar_path}: {e}")

        return images

    def _create_image_batches(self, all_image_objects: list[ImageObject]) -> list[ImageBatch]:
        """Create ImageBatch objects from a list of ImageObjects."""
        image_batches = []
        for i in range(0, len(all_image_objects), self.task_batch_size):
            batch_images = all_image_objects[i:i + self.task_batch_size]

            image_batch = ImageBatch(
                task_id=f"image_batch_{i // self.task_batch_size}",
                dataset_name="tar_files",
                data=batch_images
            )
            image_batches.append(image_batch)

        if self.verbose:
            logger.info(f"Created {len(image_batches)} ImageBatch objects with task_batch_size={self.task_batch_size}")

        return image_batches

    def process(self, task: FileGroupTask) -> list[ImageBatch]:
        """Process a FileGroupTask containing tar file paths and create ImageBatch objects."""
        tar_file_paths = task.data
        if not tar_file_paths:
            if self.verbose:
                logger.warning(f"No tar file paths in task {task.task_id}")
            return []

        # Convert string paths to pathlib.Path objects
        tar_files = [pathlib.Path(tar_path) for tar_path in tar_file_paths]

        if self.verbose:
            logger.info(f"Processing {len(tar_files)} tar files in task {task.task_id}")

        # Load all images from tar files
        all_image_objects = []

        # Add progress bar for processing tar files
        for tar_file_path in tqdm(tar_files, desc=f"Processing tar files in {task.task_id}", disable=not self.verbose):
            images = self._process_tar_file(tar_file_path)
            all_image_objects.extend(images)

        if self.verbose:
            logger.info(f"Loaded {len(all_image_objects)} images total from {len(tar_files)} tar files in task {task.task_id}")

        return self._create_image_batches(all_image_objects)
