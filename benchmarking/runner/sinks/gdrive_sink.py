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

# If this sink is not enabled this entire file will not be imported, so these
# dependencies are only needed if the user intends to enable/use this sink.
from loguru import logger
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pydrive2.files import ApiRequestError
from runner.matrix import MatrixConfig, MatrixEntry
from runner.sinks.sink import Sink


class GdriveSink(Sink):
    def __init__(self, sink_config: dict[str, Any]):
        super().__init__(sink_config)
        self.sink_config = sink_config
        self.enabled = self.sink_config.get("enabled", True)
        self.results: list[dict[str, Any]] = []
        self.session_name: str = None
        self.matrix_config: MatrixConfig = None
        self.env_dict: dict[str, Any] = None
        self.drive_folder_id: str = None
        self.service_account_file: str = None

        # Retry and progress tracking configuration
        self.max_retries: int = self.sink_config.get("max_retries", 3)
        self.retry_delay_base: float = self.sink_config.get("retry_delay_base", 2.0)

        # Metadata and organization configuration
        self.include_metadata: bool = self.sink_config.get("include_metadata", True)
        self.organize_by_date: bool = self.sink_config.get("organize_by_date", True)
        self.include_environment: bool = self.sink_config.get("include_environment", True)
        self.enhanced_naming: bool = self.sink_config.get("enhanced_naming", True)

    def initialize(self, session_name: str, matrix_config: MatrixConfig, env_dict: dict[str, Any]) -> None:
        self.session_name = session_name
        self.matrix_config = matrix_config
        self.env_dict = env_dict
        self.drive_folder_id = self.sink_config.get("drive_folder_id")
        if not self.drive_folder_id:
            msg = "GdriveSink: No drive folder ID configured"
            raise ValueError(msg)
        self.service_account_file = self.sink_config.get("service_account_file")
        if not self.service_account_file:
            msg = "GdriveSink: No service account file configured"
            raise ValueError(msg)

    def process_result(self, result_dict: dict[str, Any], matrix_entry: MatrixEntry) -> None:
        """Collect benchmark results for inclusion in the Google Drive archive."""
        if not self.enabled:
            return

        # Get benchmark-specific GDrive sink configuration
        # Note: Unlike SlackSink which uses list[str] for additional_metrics (simple metric selection),
        # GDrive sink uses dict[str, Any] for complex configuration supporting multiple data types:
        # - custom_folder_id: str (folder routing)
        # - description: str (metadata)
        # - Any custom fields: various types (extensibility)
        # This reflects the different purposes: Slack does simple metric selection,
        # while GDrive does complex archive behavior configuration.
        benchmark_config = {}
        if matrix_entry:
            benchmark_config = matrix_entry.get_sink_data(self.name)
            logger.debug(f"GdriveSink: Benchmark '{matrix_entry.name}' config: {benchmark_config}")

        # Store result with benchmark-specific configuration for finalize()
        result_entry = {
            "result": result_dict,
            "benchmark_name": matrix_entry.name if matrix_entry else result_dict.get("name", "unknown"),
            "config": benchmark_config,
            "matrix_entry": matrix_entry,
        }

        self.results.append(result_entry)
        logger.debug(
            f"GdriveSink: Collected result for '{result_entry['benchmark_name']}' (success: {result_dict.get('success', False)})"
        )

    def finalize(self) -> None:
        if self.enabled:
            if not self.results:
                logger.warning("GdriveSink: No benchmark results collected, skipping upload")
                return

            tar_path = None
            finalize_start_time = time.time()
            try:
                # Phase 1: Create archive
                logger.info(f"GdriveSink: Starting finalization for {len(self.results)} benchmark results")
                tar_path = self._tar_results_and_artifacts()

                # Phase 2: Upload with enhanced tracking
                shareable_link = self._upload_to_gdrive(tar_path)

                # Success metrics
                total_finalize_time = time.time() - finalize_start_time
                logger.success(f"GdriveSink: Successfully processed {len(self.results)} benchmark results")
                logger.info(f"GdriveSink: Total finalization time: {total_finalize_time:.1f}s")
                logger.info(f"GdriveSink: Results available at: {shareable_link}")

            except Exception as e:  # noqa: BLE001
                total_finalize_time = time.time() - finalize_start_time
                tb = traceback.format_exc()
                logger.error(f"GdriveSink: Finalization failed after {total_finalize_time:.1f}s")
                logger.error(f"GdriveSink: Error details: {e}\n{tb}")
            finally:
                if tar_path:
                    self._delete_tar_file(tar_path)

            # Log configuration summary for debugging
            logger.debug(
                f"GdriveSink: Configuration - Max retries: {self.max_retries}, "
                f"Retry delay base: {self.retry_delay_base}s, "
                f"Metadata: {self.include_metadata}, Organization: {self.organize_by_date}, "
                f"Enhanced naming: {self.enhanced_naming}"
            )

            # Log per-benchmark configuration summary
            if self.results:
                configs_with_custom = [entry for entry in self.results if entry.get("config")]
                if configs_with_custom:
                    logger.info(f"GdriveSink: {len(configs_with_custom)} benchmarks have custom configurations")
                    for entry in configs_with_custom:
                        config_summary = ", ".join(
                            [f"{k}={v}" for k, v in entry["config"].items() if k != "description"]
                        )
                        if config_summary:
                            logger.debug(
                                f"GdriveSink: Benchmark '{entry['benchmark_name']}' custom config: {config_summary}"
                            )
                else:
                    logger.debug("GdriveSink: All benchmarks using default sink configuration")
        else:
            logger.warning("GdriveSink: Not enabled, skipping post.")

    def _generate_timestamp(self) -> str:
        """Generate timestamp string for unique file naming."""
        return time.strftime("%Y-%m-%d_%H-%M-%S")

    def _generate_date_string(self) -> str:
        """Generate date string for folder organization."""
        return time.strftime("%Y-%m-%d")

    def _generate_session_metadata(self) -> dict[str, Any]:
        """Generate comprehensive session metadata for archiving."""
        # Extract results for metadata calculation
        results = [entry["result"] for entry in self.results]

        metadata = {
            "session_info": {
                "name": self.session_name,
                "timestamp": self._generate_timestamp(),
                "date": self._generate_date_string(),
                "total_results": len(results),
                "successful_results": sum(1 for r in results if r.get("success", False)),
                "failed_results": sum(1 for r in results if not r.get("success", False)),
            },
            "benchmark_summary": {
                "benchmark_names": [entry["benchmark_name"] for entry in self.results],
                "execution_times": [r.get("exec_time_s", 0) for r in results if "exec_time_s" in r],
                "total_execution_time": sum(r.get("exec_time_s", 0) for r in results if "exec_time_s" in r),
                "benchmark_configs": [entry["config"] for entry in self.results if entry["config"]],
            },
            "configuration": {
                "results_path": str(self.matrix_config.results_path),
                "artifacts_path": str(self.matrix_config.artifacts_path),
                "sink_settings": {
                    "max_retries": self.max_retries,
                    "retry_delay_base": self.retry_delay_base,
                    "include_metadata": self.include_metadata,
                    "organize_by_date": self.organize_by_date,
                },
            },
        }

        # Add environment details if enabled
        if self.include_environment and self.env_dict:
            metadata["environment"] = {
                "variables": {k: v for k, v in self.env_dict.items() if not k.startswith("_")},
                "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
                "hostname": os.environ.get("HOSTNAME", "unknown"),
                "user": os.environ.get("USER", "unknown"),
            }

        return metadata

    def _generate_enhanced_filename(self) -> str:
        """Generate enhanced filename with metadata for better organization."""
        if not self.enhanced_naming:
            return f"{self.session_name}_{self._generate_timestamp()}.tar.gz"

        timestamp = self._generate_timestamp()
        # Extract results for success calculation
        results = [entry["result"] for entry in self.results]
        successful_count = sum(1 for r in results if r.get("success", False))
        total_count = len(results)

        # Create descriptive filename
        filename_parts = [self.session_name, timestamp, f"{successful_count}of{total_count}success"]

        return "_".join(filename_parts) + ".tar.gz"

    def _get_benchmark_config(self, entry: dict, key: str, default: object | None = None) -> object | None:
        """Get benchmark-specific configuration value with fallback to default."""
        return entry.get("config", {}).get(key, default)

    def _tar_results_and_artifacts(self) -> Path:
        """Create enhanced archive with session results, artifacts, and metadata."""
        results_path = Path(self.matrix_config.results_path)
        artifacts_path = Path(self.matrix_config.artifacts_path)

        # Session-specific directories (not entire results/artifacts trees)
        session_results_path = results_path / self.session_name
        session_artifacts_path = artifacts_path / self.session_name

        # Phase 3: Enhanced filename generation
        tar_filename = self._generate_enhanced_filename()
        tar_path = results_path / tar_filename

        logger.info(f"GdriveSink: Creating archive {tar_filename}")

        with tarfile.open(tar_path, "w:gz") as tar:
            # Add session results if they exist
            if session_results_path.exists():
                tar.add(session_results_path, arcname=f"results/{session_results_path.name}")
                logger.debug(f"GdriveSink: Added session results from {session_results_path}")
            else:
                logger.warning(f"GdriveSink: Session results directory not found: {session_results_path}")

            # Add session artifacts if they exist
            if session_artifacts_path.exists():
                tar.add(session_artifacts_path, arcname=f"artifacts/{session_artifacts_path.name}")
                logger.debug(f"GdriveSink: Added session artifacts from {session_artifacts_path}")
            else:
                logger.debug(f"GdriveSink: No session artifacts directory found: {session_artifacts_path}")

            # Phase 3: Add session metadata if enabled
            if self.include_metadata:
                metadata = self._generate_session_metadata()
                metadata_json = json.dumps(metadata, indent=2, default=str)

                # Create temporary metadata file
                metadata_filename = f"{self.session_name}_metadata.json"
                metadata_path = results_path / metadata_filename

                try:
                    with open(metadata_path, "w") as f:
                        f.write(metadata_json)

                    tar.add(metadata_path, arcname=f"metadata/{metadata_filename}")
                    logger.debug(f"GdriveSink: Added session metadata ({len(metadata_json)} bytes)")

                finally:
                    # Clean up temporary metadata file
                    if metadata_path.exists():
                        metadata_path.unlink()

        # Log archive size and contents
        archive_size_mb = tar_path.stat().st_size / (1024 * 1024)
        logger.info(f"GdriveSink: Archive created: {archive_size_mb:.2f} MB")
        if self.include_metadata:
            logger.debug(f"GdriveSink: Archive includes metadata for {len(self.results)} benchmark results")

        return tar_path

    def _create_gdrive_folder_structure(self, drive: GoogleDrive) -> str:
        """Create organized folder structure in Google Drive and return target folder ID."""
        if not self.organize_by_date:
            return self.drive_folder_id

        try:
            date_string = self._generate_date_string()

            # Check if date folder already exists
            query = f"'{self.drive_folder_id}' in parents and name='{date_string}' and mimeType='application/vnd.google-apps.folder'"
            existing_folders = drive.ListFile({"q": query}).GetList()

            if existing_folders:
                folder_id = existing_folders[0]["id"]
                logger.debug(f"GdriveSink: Using existing date folder: {date_string} (ID: {folder_id})")
            else:
                # Create new date folder
                date_folder = drive.CreateFile(
                    {
                        "title": date_string,
                        "parents": [{"id": self.drive_folder_id}],
                        "mimeType": "application/vnd.google-apps.folder",
                    }
                )
                date_folder.Upload()
                folder_id = date_folder["id"]
                logger.info(f"GdriveSink: Created new date folder: {date_string} (ID: {folder_id})")

        except ApiRequestError as e:
            logger.warning(f"GdriveSink: Failed to create folder structure, using root folder: {e}")
            return self.drive_folder_id
        else:
            return folder_id

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

                # Phase 3: Create organized folder structure
                # Check for benchmark-specific folder overrides
                custom_folder_id = None
                if self.results:
                    # Check if any benchmark specifies a custom folder
                    for entry in self.results:
                        entry_folder_id = self._get_benchmark_config(entry, "custom_folder_id")
                        if entry_folder_id:
                            custom_folder_id = entry_folder_id
                            logger.info(
                                f"GdriveSink: Using custom folder ID from benchmark '{entry['benchmark_name']}': {custom_folder_id}"
                            )
                            break

                if custom_folder_id:
                    target_folder_id = custom_folder_id
                else:
                    target_folder_id = self._create_gdrive_folder_structure(drive)

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

                # Phase 3: Enhanced logging with organization info
                if self.organize_by_date:
                    logger.info(f"GdriveSink: File organized in date folder: {self._generate_date_string()}")
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
