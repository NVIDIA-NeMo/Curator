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

    def process(self, _: _EmptyTask) -> list[ImageBatch]:
        """Process webdataset tar files and create ImageBatch objects."""
        if self.input_dataset_path is None:
            msg = "input_dataset_path is not set"
            raise ValueError(msg)
        
        # Find all tar files in the dataset directory
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

        # Load all images from tar files
        all_image_objects = []
        total_images_loaded = 0

        for tar_path in tar_files:
            if isinstance(tar_path, str):
                tar_path = pathlib.Path(tar_path)
            
            if self.verbose:
                logger.info(f"Processing tar file: {tar_path}")
            
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    # Get all image files (jpg) in the tar
                    image_members = [m for m in tar.getmembers() if m.name.endswith('.jpg') and m.isfile()]
                    
                    for member in image_members:
                        # Check image limit
                        if self.image_limit > 0 and total_images_loaded >= self.image_limit:
                            break
                        
                        try:
                            # Extract image key from filename (remove .jpg extension)
                            image_key = member.name.replace('.jpg', '')
                            
                            # Load image from tar
                            image_file = tar.extractfile(member)
                            if image_file is None:
                                if self.verbose:
                                    logger.warning(f"Could not extract image {member.name} from {tar_path}")
                                continue
                                
                            # Load image with PIL and convert to RGB numpy array
                            with Image.open(image_file) as img:
                                img_rgb = img.convert("RGB")
                                image_data = np.array(img_rgb, dtype=np.uint8)
                            
                            # Create ImageObject with loaded data
                            image_obj = ImageObject(
                                image_path=str(tar_path / member.name),  # Virtual path in tar
                                image_id=image_key,
                                image_data=image_data
                            )
                            
                            all_image_objects.append(image_obj)
                            total_images_loaded += 1
                            
                        except Exception as e:
                            if self.verbose:
                                logger.error(f"Error loading image {member.name} from {tar_path}: {e}")
                            continue
                    
                    # Break if we hit the image limit
                    if self.image_limit > 0 and total_images_loaded >= self.image_limit:
                        break
                        
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error processing tar file {tar_path}: {e}")
                continue

        if self.verbose:
            logger.info(f"Loaded {total_images_loaded} images total from {len(tar_files)} tar files")

        if self.image_limit > 0 and total_images_loaded > self.image_limit:
            all_image_objects = all_image_objects[:self.image_limit]
            if self.verbose:
                logger.info(f"Limiting to first {self.image_limit} images")

        # Group images into ImageBatch objects
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
