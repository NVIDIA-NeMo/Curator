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

import json
import os
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any

from runner.entry import Entry
from runner.session import Session
from runner.sinks.sink import Sink

# If this sink is not enabled this entire file will not be imported, so these
# dependencies are only needed if the user intends to enable/use this sink.
from loguru import logger
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pydrive2.files import ApiRequestError

class GdriveSink(Sink):
    def __init__(self, sink_config: dict[str, Any]):
        super().__init__(sink_config)
        self.sink_config = sink_config
        self.name = sink_config.get("name", "gdrive")
        self.enabled = self.sink_config.get("enabled", True)
        self.session_name: str = None
        self.session: Session = None
        self.env_dict: dict[str, Any] = None
        self.drive_folder_id: str = None
        self.service_account_file: str = None

        self.max_retries: int = self.sink_config.get("max_retries", 3)
        self.retry_delay_base: float = self.sink_config.get("retry_delay_base", 2.0)

    def initialize(self, session_name: str, session: Session, env_dict: dict[str, Any]) -> None:
        self.session_name = session_name
        self.session = session
        self.env_dict = env_dict
        self.drive_folder_id = self.sink_config.get("drive_folder_id")
        if not self.drive_folder_id:
            msg = "GdriveSink: No drive folder ID configured"
            raise ValueError(msg)
        self.service_account_file = self.sink_config.get("service_account_file")
        if not self.service_account_file:
            msg = "GdriveSink: No service account file configured"
            raise ValueError(msg)

    def process_result(self, result_dict: dict[str, Any], entry: Entry) -> None:
        pass

    def finalize(self) -> None:
        if self.enabled:
            tar_path = None
            try:
                #tar_path = self._tar_results()
                tar_path=Path("/tmp/test.html")
                shareable_link = self._upload_to_gdrive(tar_path)
                logger.info(f"GdriveSink: Results available at: {shareable_link}")
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                logger.error(f"GdriveSink: Error details: {e}\n{tb}")
            finally:
                if tar_path:
                    self._delete_tar_file(tar_path)
        else:
            logger.warning("GdriveSink: Not enabled, skipping.")

    def _tar_results(self) -> Path:
        """Create archive with session results."""
        tar_path = None
        results_path = Path(self.session.results_path)
        session_results_path = results_path / self.session_name
        
        if session_results_path.exists():
            tar_filename = f"{self.session_name}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.tar.gz"
            tar_path = results_path / tar_filename
        
            logger.info(f"GdriveSink: Creating archive {tar_filename}")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(session_results_path, arcname=f"results/{session_results_path.name}")

            archive_size_mb = tar_path.stat().st_size / (1024 * 1024)
            logger.info(f"GdriveSink: Archive created: {archive_size_mb:.2f} MB")
        
        else:
            logger.warning(f"GdriveSink: Session results directory not found: {session_results_path}")

        return tar_path

    def _upload_to_gdrive(self, tar_path: Path) -> str:
        """Upload archive to Google Drive with enhanced organization and retry logic."""
        file_size_mb = tar_path.stat().st_size / (1024 * 1024)
        logger.info(f"GdriveSink: Starting upload to Google Drive: {tar_path.name} ({file_size_mb:.2f} MB)")

        # Track overall upload timing
        upload_start_time = time.time()

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"GdriveSink: Upload attempt {attempt}/{self.max_retries}")

                # Initialize Google Drive client
                gauth = GoogleAuth()
                scope = ["https://www.googleapis.com/auth/drive"]
                gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(self.service_account_file, scope)
                drive = GoogleDrive(gauth)

                # Create folder structure
                target_folder_id = self._create_gdrive_folder(drive)

                # Create file and upload
                attempt_start_time = time.time()
                drive_file = drive.CreateFile({"parents": [{"id": target_folder_id}], "title": tar_path.name})
                drive_file.SetContentFile(tar_path)
                drive_file.Upload()

                # Calculate upload metrics
                attempt_duration = time.time() - attempt_start_time
                total_duration = time.time() - upload_start_time
                upload_speed_mbps = file_size_mb / attempt_duration if attempt_duration > 0 else 0

                shareable_link = drive_file["alternateLink"]
                logger.success(f"GdriveSink: Upload completed successfully on attempt {attempt}!")
                logger.info(
                    f"GdriveSink: Upload metrics - Duration: {attempt_duration:.1f}s, Speed: {upload_speed_mbps:.2f} MB/s"
                )
                logger.info(f"GdriveSink: Total time (including retries): {total_duration:.1f}s")
                logger.info(f"GdriveSink: Shareable link: {shareable_link}")

                return shareable_link  # noqa: TRY300

            except (ApiRequestError, OSError, ValueError) as e:
                error_type = self._categorize_error(e)
                logger.warning(f"GdriveSink: Upload attempt {attempt} failed - {error_type}: {e}")

                if attempt == self.max_retries:
                    # Final attempt failed
                    total_duration = time.time() - upload_start_time
                    logger.error(
                        f"GdriveSink: All {self.max_retries} upload attempts failed after {total_duration:.1f}s"
                    )
                    raise

                # Calculate retry delay with exponential backoff
                retry_delay = self._calculate_retry_delay(attempt, error_type)
                logger.info(f"GdriveSink: Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)

        # This should never be reached, but just in case
        msg = "GdriveSink: Upload failed after all retry attempts"
        raise RuntimeError(msg)

    def _create_gdrive_folder(self, drive: GoogleDrive) -> str:
        """Create organized folder structure in Google Drive and return target folder ID."""
        try:
            # Check if folder already exists
            query = f"'{self.drive_folder_id}' in parents and title = '{self.session_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            existing_folders = drive.ListFile({"q": query}).GetList()

            if existing_folders:
                folder_id = existing_folders[0]["id"]
                logger.debug(f"GdriveSink: Using existing folder: {self.session_name} (ID: {folder_id})")
            else:
                # Create new date folder
                date_folder = drive.CreateFile(
                    {
                        "title": self.session_name,
                        "parents": [{"id": self.drive_folder_id}],
                        "mimeType": "application/vnd.google-apps.folder",
                    }
                )
                date_folder.Upload()
                folder_id = date_folder["id"]
                logger.info(f"GdriveSink: Created new folder: {self.session_name} (ID: {folder_id})")

        except ApiRequestError as e:
            logger.warning(f"GdriveSink: Failed to create folder, using root folder: {e}")
            return self.drive_folder_id
        else:
            return folder_id

    def _categorize_error(self, error: Exception) -> str:
        """Categorize errors for better retry logic and user feedback."""
        error_str = str(error).lower()

        # Rate limiting or quota errors
        if any(keyword in error_str for keyword in ["quota", "rate", "limit", "429"]):
            return "Rate/Quota Limit"

        # Network/connectivity errors
        if any(keyword in error_str for keyword in ["network", "timeout", "connection", "unreachable"]):
            return "Network Error"

        # Authentication errors
        if any(keyword in error_str for keyword in ["auth", "credential", "permission", "401", "403"]):
            return "Authentication Error"

        # Server errors (potentially retryable)
        if any(keyword in error_str for keyword in ["server", "500", "502", "503", "504"]):
            return "Server Error"

        # Generic API errors
        if isinstance(error, ApiRequestError):
            return "API Request Error"

        return "Unknown Error"

    def _calculate_retry_delay(self, attempt: int, error_type: str) -> float:
        """Calculate retry delay with exponential backoff and error-specific adjustments."""
        base_delay = self.retry_delay_base**attempt

        # Adjust delay based on error type
        if error_type == "Rate/Quota Limit":
            # Longer delays for rate limiting
            base_delay *= 3.0
        elif error_type == "Server Error":
            # Moderate delays for server issues
            base_delay *= 1.5
        elif error_type == "Authentication Error":
            # Short delay for auth errors (likely won't help, but try once)
            base_delay *= 0.5

        # Cap maximum delay at 60 seconds
        return min(base_delay, 60.0)

    def _delete_tar_file(self, tar_path: Path) -> None:
        """Clean up temporary archive file."""
        if tar_path.exists():
            try:
                tar_path.unlink()
                logger.debug(f"GdriveSink: Cleaned up temporary file: {tar_path.name}")
            except OSError as e:
                logger.warning(f"GdriveSink: Failed to delete temporary file {tar_path.name}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GdriveSink command line arguments")
    parser.add_argument("--drive-folder-id", type=str, required=True, help="Google Drive folder ID")
    parser.add_argument("--service-account-file", type=str, required=True, help="Path to Google Service Account JSON")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing results to archive/upload")

    args = parser.parse_args()

    sink_config = {
        "name": "gdrive",
        "enabled": True,
        "drive_folder_id": args.drive_folder_id,
        "service_account_file": args.service_account_file,
        "max_retries": 3,
        "retry_delay_base": 2.0,
    }
    results_path = Path(args.results_dir)
    session_name = results_path.name
    results_path = results_path.parent
    
    session = Session(results_path=results_path, artifacts_path=None)
    sink = GdriveSink(sink_config)
    sink.initialize(session_name=session_name, session=session, env_dict={})
    sink.finalize()