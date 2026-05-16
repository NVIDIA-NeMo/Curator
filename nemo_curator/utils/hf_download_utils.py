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

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def download_model_from_hf(
    model_id: str,
    local_dir: str | Path,
    ignore_patterns: list[str] | None = None,
    filename: str | None = None,
    revision: str | None = None,
) -> None:
    """Download a model from Hugging Face.

    This function downloads either a specific file or the entire model repository
    from Hugging Face Hub to a local directory.

    Args:
        model_id (str): The Hugging Face model identifier (e.g., 'gpt2', 'bert-base-uncased')
        local_dir (str | Path): Local directory where the model will be downloaded
        ignore_patterns (list[str] | None, optional): List of glob patterns to ignore when downloading.
            Only used when filename is not provided. Defaults to None.
        filename (str | None, optional): Specific file to download from the repository.
            If provided, only this file will be downloaded and ignore_patterns will be ignored.
            Defaults to None.
        revision (str | None, optional): Git revision (branch, tag, or commit hash) to download.
            Defaults to None (latest main branch).

    Raises:
        ValueError: If both filename and ignore_patterns are provided (not supported).

    Examples:
        # Download entire model repository
        download_model_from_hf('gpt2', './models/gpt2')

        # Download specific file
        download_model_from_hf('gpt2', './models/gpt2', filename='config.json')

        # Download with ignore patterns
        download_model_from_hf('gpt2', './models/gpt2',
                              ignore_patterns=['*.bin', '*.safetensors'])

        # Download specific revision
        download_model_from_hf('gpt2', './models/gpt2', revision='main')
    """
    if filename:
        if ignore_patterns:
            msg = "ignore_patterns is not supported when filename is provided"
            raise ValueError(msg)
        hf_hub_download(
            repo_id=model_id,
            local_dir=local_dir,
            cache_dir=local_dir,
            filename=filename,
            revision=revision,
        )
    else:
        snapshot_download(
            repo_id=model_id,
            cache_dir=local_dir,
            local_dir=local_dir,
            ignore_patterns=ignore_patterns,
            revision=revision,
        )
