import sys

from loguru import logger

from nemo_curator.stages.interleaved.io.readers.webdataset import WebdatasetReaderStage
from nemo_curator.tasks import FileGroupTask

LOCAL_TAR = "/opt/Curator/omni_data/omnicorpus-000104.tar"

logger.remove()
logger.add(sys.stderr, level="ERROR")

task = FileGroupTask(task_id="quickstart", dataset_name="mint1t", data=[LOCAL_TAR], _metadata={})
reader = WebdatasetReaderStage(
    source_id_field="pdf_name",
    materialize_on_read=True,
    per_image_fields=["image_metadata"],
)
result = reader.process(task)
batch = result[0] if isinstance(result, list) else result

print(f"Samples: {batch.num_items}")
print(f"Total rows: {batch.count()}")
print(f"  text:     {batch.count(modality='text')}")
print(f"  image:    {batch.count(modality='image')}")
print(f"  metadata: {batch.count(modality='metadata')}")
