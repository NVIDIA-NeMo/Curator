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

import csv
import io
import json
import pathlib
import uuid
from typing import Any

import pandas as pd
from loguru import logger


class JsonEncoderCustom(json.JSONEncoder):
    """Custom JSON encoder that handles types that are not JSON serializable.

    Example:
    ```python
    json.dumps(data, cls=JsonEncoderClass)
    ```

    """

    def default(self, obj: object) -> str | object:
        """Encode an object for JSON serialization.

        Args:
            obj: Object to encode.

        Returns:
            Encoded object.

        """
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)  # type: ignore[no-any-return]


def write_bytes(  # noqa: PLR0913
    buffer: bytes,
    dest: pathlib.Path,
    desc: str,
    source_video: str,
    *,
    verbose: bool,
    backup_and_overwrite: bool = False,
    overwrite: bool = False,
) -> None:
    """Write bytes to local path.

    Args:
        buffer: Bytes to write.
        dest: Destination to write.
        desc: Description of the write.
        source_video: Source video.
        verbose: Verbosity.
        client: Storage client.
        backup_and_overwrite: Backup and overwrite.
        overwrite: Overwrite.

    """
    if dest.exists():
        if backup_and_overwrite:
            msg = "Backup and overwrite is not implemented"
            raise NotImplementedError(msg)
        elif overwrite:
            logger.warning(f"{desc} {dest} already exists, overwriting ...")
        else:
            logger.warning(f"{desc} {dest} already exists, skipping ...")
            return
    if verbose:
        logger.info(f"Writing {desc} for {source_video} to {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fp:
        fp.write(buffer)


def write_parquet(  # noqa: PLR0913
    data: list[dict[str, str]],
    dest: pathlib.Path,
    desc: str,
    source_video: str,
    *,
    verbose: bool,
    backup_and_overwrite: bool = False,
    overwrite: bool = False,
) -> None:
    """Write parquet to local path.

    Args:
        data: Data to write.
        dest: Destination to write.
        desc: Description of the write.
        source_video: Source video.
        verbose: Verbosity.
        client: Storage client.
        backup_and_overwrite: Whether to backup existing file before overwriting.
        overwrite: Whether to overwrite existing file.

    """
    # Convert list of dicts to DataFrame
    pdf = pd.DataFrame(data)

    # Write DataFrame to a Parquet file in memory
    parquet_buffer = io.BytesIO()
    pdf.to_parquet(parquet_buffer, index=False)

    write_bytes(
        parquet_buffer.getvalue(),
        dest,
        desc,
        source_video,
        verbose=verbose,
        backup_and_overwrite=backup_and_overwrite,
        overwrite=overwrite,
    )


def write_json(  # noqa: PLR0913
    data: dict[str, Any],
    dest: pathlib.Path,
    desc: str,
    source_video: str,
    *,
    verbose: bool,
    backup_and_overwrite: bool = False,
    overwrite: bool = False,
) -> None:
    """Write json to local path.

    Args:
        data: Data to write.
        dest: Destination to write.
        desc: Description of the write.
        source_video: Source video.
        verbose: Verbosity.
        client: Storage client.
        backup_and_overwrite: Backup and overwrite.
        overwrite: Overwrite.

    """
    updated_json = json.dumps(data, indent=4, cls=JsonEncoderCustom)
    json_bytes = io.BytesIO(updated_json.encode("utf-8"))
    write_bytes(
        json_bytes.getvalue(),
        dest,
        desc,
        source_video,
        verbose=verbose,
        backup_and_overwrite=backup_and_overwrite,
        overwrite=overwrite,
    )


def write_csv(  # noqa: PLR0913
    dest: pathlib.Path,
    desc: str,
    source_video: str,
    data: list[list[str]],
    *,
    verbose: bool,
    backup_and_overwrite: bool = False,
) -> None:
    """Write csv to local path.

    Args:
        dest: Destination to write.
        desc: Description of the write.
        source_video: Source video.
        data: Data to write.
        verbose: Verbosity.
        client: Storage client.
        backup_and_overwrite: Backup and overwrite.

    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    csv_bytes = io.BytesIO(output.getvalue().encode("utf-8"))
    write_bytes(
        csv_bytes.getvalue(),
        dest,
        desc,
        source_video,
        verbose=verbose,
        backup_and_overwrite=backup_and_overwrite,
    )
