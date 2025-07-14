import subprocess
from urllib.parse import urlparse

from loguru import logger

from ray_curator.stages.download.text import DocumentDownloader


class WikipediaDownloader(DocumentDownloader):
    """Downloads Wikipedia dump files (.bz2) from wikimedia.org."""

    def __init__(self, download_dir: str, verbose: bool = False):
        """
        Creates a Wikipedia downloader.

        Args:
            download_dir: Path to store raw compressed .bz2 files
            verbose: If True, logs stdout and stderr of the download command
        """
        super().__init__(download_dir, verbose)

    def _get_output_filename(self, url: str) -> str:
        """Generate output filename from URL."""
        urlpath = urlparse(url).path[1:]
        return urlpath.replace("/", "-")

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        """Download a Wikipedia dump file to the specified path.

        Args:
            url: URL to download
            path: Local path to save file

        Returns:
            Tuple of (success, error_message). If success is True, error_message is None.
            If success is False, error_message contains the error details.
        """
        if self._verbose:
            logger.info(f"Downloading {url} to {path}")

        # Download with wget
        cmd = ["wget", url, "-O", path]

        # Set up stdout/stderr handling
        if self._verbose:
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.DEVNULL, subprocess.PIPE

        result = subprocess.run(  # noqa: S603, PLW1510
            cmd,
            stdout=stdout,
            stderr=stderr,
        )

        if result.returncode == 0:
            return True, None
        else:
            error_msg = result.stderr.decode("utf-8") if result.stderr else "Unknown error"
            return False, error_msg
