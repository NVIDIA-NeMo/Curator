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

import tarfile
import traceback
from pathlib import Path
from typing import Any

# If this sink is not enabled this entire file will not be imported, so these
# dependencies are only needed if the user intends to enable/use this sink.
from loguru import logger
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from runner.entry import Entry
from runner.session import Session
from runner.sinks.sink import Sink


class GdriveSink(Sink):
    def __init__(self, sink_config: dict[str, Any]):
        super().__init__(sink_config)
        self.sink_config = sink_config
        self.results: list[dict[str, Any]] = []
        self.session_name: str = None
        self.matrix_config: Session = None
        self.env_dict: dict[str, Any] = None
        self.drive_folder_id: str = None
        self.service_account_file: str = None

    def initialize(self, session_name: str, matrix_config: Session, env_dict: dict[str, Any]) -> None:
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

    def process_result(self, result_dict: dict[str, Any], matrix_entry: Entry) -> None:
        pass

    def finalize(self) -> None:
        try:
            tar_path = self._tar_results_and_artifacts()
            self._upload_to_gdrive(tar_path)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.error(f"GdriveSink: Error uploading to Google Drive: {e}\n{tb}")
        finally:
            self._delete_tar_file(tar_path)

    def _tar_results_and_artifacts(self) -> Path:
        results_path = Path(self.matrix_config.results_path)
        artifacts_path = Path(self.matrix_config.artifacts_dir)
        tar_path = results_path / f"{self.session_name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(results_path, arcname=results_path.name)
            tar.add(artifacts_path, arcname=artifacts_path.name)
        return tar_path

    def _upload_to_gdrive(self, tar_path: Path) -> None:
        gauth = GoogleAuth()
        scope = ["https://www.googleapis.com/auth/drive"]
        gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(self.service_account_file, scope)
        drive = GoogleDrive(gauth)

        drive_file = drive.CreateFile({"parents": [{"id": self.drive_folder_id}], "title": tar_path.name})
        drive_file.SetContentFile(tar_path)
        drive_file.Upload()
        return drive_file["alternateLink"]

    def _delete_tar_file(self, tar_path: Path) -> None:
        if tar_path.exists():
            tar_path.unlink()
