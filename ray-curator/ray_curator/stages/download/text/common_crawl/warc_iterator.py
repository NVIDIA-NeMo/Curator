from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger

# TODO: Consider using fastwarc https://github.com/NVIDIA-NeMo/Curator/issues/778
from warcio.archiveiterator import ArchiveIterator

from ray_curator.stages.download.text import DocumentIterator


class CommonCrawlWarcIterator(DocumentIterator):
    """Processes downloaded WARC files."""

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Process a task containing WARC files and extract their contents."""
        filename = file_path.name if isinstance(file_path, Path) else file_path.split("/")[-1]

        num_records = 0
        # TODO: Support cloud storage https://github.com/NVIDIA-NeMo/Curator/issues/779
        with open(file_path, "rb") as file_pointer:
            archive_iterator = ArchiveIterator(file_pointer, arc2warc=True)
            while True:
                try:
                    rec = next(archive_iterator)
                    if rec.rec_type == "response":
                        content = rec.content_stream().read()
                        warc_id = rec.rec_headers.get_header("WARC-Record-ID")[10:-1]
                        url = rec.rec_headers.get_header("WARC-Target-URI")
                        yield {"url": url, "warc_id": warc_id, "source_id": filename, "content": content}
                        num_records += 1
                except StopIteration:  # noqa: PERF203
                    # End of file reached normally
                    break
                except Exception as e:  # noqa: BLE001
                    # Handle corruption or other errors
                    logger.error(f"Error processing record {num_records} in {filename}: {e!s}")
                    # Try to continue with next record
                    continue

    def output_columns(self) -> list[str]:
        return ["url", "warc_id", "source_id", "content"]
