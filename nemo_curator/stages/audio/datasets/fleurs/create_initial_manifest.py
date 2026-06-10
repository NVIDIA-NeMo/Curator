# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
import shutil
from dataclasses import dataclass
from typing import Any

from huggingface_hub import hf_hub_download
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.datasets.file_utils import extract_archive
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, _EmptyTask

# Hugging Face dataset repo hosting FLEURS.
FLEURS_HF_REPO_ID = "google/fleurs"


def get_fleurs_filenames(lang: str, split: str) -> tuple[str, str]:
    """Return the repo-relative (transcript_tsv, audio_archive) paths in ``google/fleurs``.

    examples
    "data/hy_am/dev.tsv"
    "data/hy_am/audio/dev.tar.gz"
    """
    tsv_filename = f"data/{lang}/{split}.tsv"
    audio_filename = f"data/{lang}/audio/{split}.tar.gz"
    return tsv_filename, audio_filename


@dataclass
class CreateInitialManifestFleursStage(ProcessingStage[_EmptyTask, AudioTask]):
    """Create initial manifest for the FLEURS dataset.

    Dataset link: https://huggingface.co/datasets/google/fleurs

    Downloads all files, extracts them, and emits one ``AudioTask`` per
    transcript line keyed by ``filepath_key`` and ``text_key``.

    Args:
        lang: Language code (e.g. ``"hy_am"`` for Armenian).
        split: Dataset split (``"test"``, ``"train"``, or ``"dev"``).
        raw_data_dir: Folder for extracting the audio archive. When
            ``auto_download=False`` this is instead the pre-staged dataset dir
            containing ``<split>.tsv`` and ``<split>/`` (extracted ``.wav`` files).
        filepath_key: Key name used for the audio file path in each emitted entry.
        text_key: Key name used for the transcript text in each emitted entry.
        cache_dir: Optional Hugging Face cache directory for the downloaded files.
            When ``None`` the default Hugging Face cache (``HF_HOME``) is used.
            Only used on the one-time download path.
        auto_download: Controls behavior only when ``raw_data_dir`` is not already
            populated. When ``True`` (default) the dataset is fetched from Hugging
            Face exactly once and staged into ``raw_data_dir`` (transcript +
            extracted audio); every subsequent run finds it on disk and performs no
            network I/O. When ``False`` the stage never downloads and requires the
            dataset to be pre-staged (e.g. by
            ``benchmarking/data_prep/prepare_fleurs_data.py``).
    """

    name: str = "CreateInitialManifestFleurs"
    lang: str = ""
    split: str = ""
    raw_data_dir: str = ""
    filepath_key: str = "audio_filepath"
    text_key: str = "text"
    batch_size: int = 1
    cache_dir: str | None = None
    auto_download: bool = True

    def __post_init__(self) -> None:
        for attr in ("lang", "split", "raw_data_dir"):
            if not getattr(self, attr):
                msg = f"{attr} is required for CreateInitialManifestFleursStage"
                raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key, self.text_key]

    def process_transcript(self, file_path: str, audio_root: str) -> list[AudioTask]:
        """Parse transcript TSV file and emit one AudioTask per line.

        Args:
            file_path: Path to the transcript ``.tsv`` file.
            audio_root: Directory containing the extracted ``.wav`` files.
        """
        entries: list[AudioTask] = []
        min_num_parts = 2
        with open(file_path, encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) < min_num_parts:
                    continue

                file_name, transcript_text = parts[1], parts[2]
                abs_wav = os.path.abspath(os.path.join(audio_root, file_name))

                entries.append(
                    AudioTask(
                        data={self.filepath_key: abs_wav, self.text_key: transcript_text},
                        dataset_name=f"Fleurs_{self.lang}_{self.split}_{self.raw_data_dir}",
                        filepath_key=self.filepath_key,
                    )
                )
        return entries

    def _prestaged_paths(self, dst_folder: str) -> tuple[str, str]:
        """Return the expected ``(transcript_tsv, audio_root)`` paths under ``dst_folder``."""
        return os.path.join(dst_folder, f"{self.split}.tsv"), os.path.join(dst_folder, self.split)

    def is_prestaged(self, dst_folder: str) -> bool:
        """True when the dataset is already staged on disk (transcript + audio present)."""
        tsv_path, audio_root = self._prestaged_paths(dst_folder)
        return os.path.isfile(tsv_path) and os.path.isdir(audio_root)

    def download_extract_files(self, dst_folder: str) -> tuple[str, str]:
        """Download the FLEURS transcript + audio archive once and stage them in ``dst_folder``.

        Uses ``huggingface_hub.hf_hub_download`` (which retries transient HTTP
        errors including 429 with backoff). The audio archive is extracted into
        ``<dst_folder>/<split>/`` and the transcript is copied to
        ``<dst_folder>/<split>.tsv`` so subsequent runs find the dataset on disk
        and skip the download entirely.

        Returns:
            Tuple of ``(transcript_tsv_path, audio_root)`` where both live under
            ``dst_folder``.
        """
        os.makedirs(dst_folder, exist_ok=True)

        tsv_filename, audio_filename = get_fleurs_filenames(self.lang, self.split)
        hf_tsv_path = hf_hub_download(
            repo_id=FLEURS_HF_REPO_ID,
            repo_type="dataset",
            filename=tsv_filename,
            cache_dir=self.cache_dir,
        )
        archive_path = hf_hub_download(
            repo_id=FLEURS_HF_REPO_ID,
            repo_type="dataset",
            filename=audio_filename,
            cache_dir=self.cache_dir,
        )

        extract_archive(archive_path, str(dst_folder), force_extract=True)

        # Stage the transcript next to the extracted audio so the dataset is
        # self-contained on disk and reused (no re-download) on the next run.
        staged_tsv_path, audio_root = self._prestaged_paths(dst_folder)
        if os.path.abspath(hf_tsv_path) != os.path.abspath(staged_tsv_path):
            shutil.copyfile(hf_tsv_path, staged_tsv_path)
        return staged_tsv_path, audio_root

    def locate_prestaged_files(self, dst_folder: str) -> tuple[str, str]:
        """Locate a pre-staged FLEURS transcript + extracted audio (no download).

        Expects the on-disk layout produced either by a prior auto-download run or
        by ``benchmarking/data_prep/prepare_fleurs_data.py``:
        ``<dst_folder>/<split>.tsv`` (transcript) and ``<dst_folder>/<split>/``
        (extracted ``.wav`` files).

        Returns:
            Tuple of ``(transcript_tsv_path, audio_root)``.
        """
        tsv_path, audio_root = self._prestaged_paths(dst_folder)
        if not os.path.isfile(tsv_path):
            msg = (
                f"Pre-staged FLEURS transcript not found at {tsv_path}. Run "
                "benchmarking/data_prep/prepare_fleurs_data.py to stage the dataset, "
                "or set auto_download=True."
            )
            raise FileNotFoundError(msg)
        if not os.path.isdir(audio_root):
            msg = (
                f"Pre-staged FLEURS audio directory not found at {audio_root}. Run "
                "benchmarking/data_prep/prepare_fleurs_data.py to stage the dataset, "
                "or set auto_download=True."
            )
            raise FileNotFoundError(msg)
        return tsv_path, audio_root

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, _: _EmptyTask) -> list[AudioTask]:
        # Auto-download only ever happens when the dataset has never been
        # downloaded: if it is already staged on disk, always reuse it.
        if self.is_prestaged(self.raw_data_dir):
            logger.info(f"Reusing pre-staged FLEURS dataset at {self.raw_data_dir} (no download)")
            tsv_path, audio_root = self._prestaged_paths(self.raw_data_dir)
        elif self.auto_download:
            logger.info(f"FLEURS dataset not found at {self.raw_data_dir}; downloading once")
            tsv_path, audio_root = self.download_extract_files(self.raw_data_dir)
        else:
            tsv_path, audio_root = self.locate_prestaged_files(self.raw_data_dir)
        return self.process_transcript(tsv_path, audio_root)
