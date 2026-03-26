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

import glob
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from loguru import logger

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.audio.datasets.file_utils import download_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, _EmptyTask

SAMPLE_RATE_48KHZ = 48000

DNS_READSPEECH_URL = (
    "https://dnschallengepublic.blob.core.windows.net/dns5archive/"
    "V5_training_dataset/Track1_Headset/read_speech.tgz.partaa"
)


@dataclass
class CreateInitialManifestReadSpeechStage(ProcessingStage[_EmptyTask, AudioTask]):
    """
    Stage to create initial manifest for the DNS Challenge Read Speech dataset.

    Dataset: Microsoft DNS Challenge 5 - Read Speech (Track 1 Headset)
    Source: https://github.com/microsoft/DNS-Challenge

    Downloads a single archive (4.88 GB) containing 14,279 WAV files at 48kHz (19.3 hours).
    When ``auto_download=True``, the archive is downloaded and extracted automatically.

    Args:
        raw_data_dir: Directory where data will be downloaded/extracted to.
        max_samples: Maximum number of samples to include (-1 for all).
        auto_download: If True, automatically download and extract dataset.
    """

    raw_data_dir: str
    max_samples: int = 5000
    auto_download: bool = True
    filepath_key: str = "audio_filepath"
    text_key: str = "text"
    name: str = "CreateInitialManifestReadSpeech"
    batch_size: int = 1

    def __post_init__(self):
        super().__init__()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key, self.text_key]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def download_and_extract(self) -> str:
        """Download and extract DNS Challenge Read Speech dataset (~4.88 GB)."""
        os.makedirs(self.raw_data_dir, exist_ok=True)

        existing_dir = self._find_extracted_wavs(self.raw_data_dir)
        if existing_dir:
            wav_count = self._count_wavs_recursive(existing_dir)
            logger.info(f"Dataset already extracted at {existing_dir} ({wav_count} WAV files)")
            return existing_dir

        logger.info("=" * 60)
        logger.info("DNS Challenge 5 - Read Speech Download")
        logger.info(f"Downloading to: {self.raw_data_dir}")
        logger.info("=" * 60)

        filename = "read_speech.tgz.partaa"
        filepath = os.path.join(self.raw_data_dir, filename)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.info(f"Archive already downloaded: {os.path.getsize(filepath) / (1024**3):.2f} GB")
        else:
            if os.path.exists(filepath):
                os.remove(filepath)
            filepath = download_file(DNS_READSPEECH_URL, self.raw_data_dir, verbose=True)
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                os.remove(filepath)
                msg = "Download failed - empty file"
                raise RuntimeError(msg)
            logger.info(f"Downloaded: {file_size / (1024**3):.2f} GB")

        logger.info("Extracting archive...")
        self._extract_archive(filepath, self.raw_data_dir)

        extracted_dir = self._find_extracted_wavs(self.raw_data_dir)
        if not extracted_dir:
            msg = "Extraction failed - no WAV files found"
            raise RuntimeError(msg)

        wav_count = self._count_wavs_recursive(extracted_dir)
        logger.info(f"Extraction complete: {wav_count} WAV files in {extracted_dir}")

        os.remove(filepath)
        logger.info(f"Removed archive: {filename}")

        logger.info("=" * 60)
        logger.info(f"Dataset ready: {wav_count} WAV files")
        logger.info(f"  Location: {extracted_dir}")
        logger.info("=" * 60)

        return extracted_dir

    def _find_extracted_wavs(self, search_dir: str) -> str | None:
        if not os.path.exists(search_dir):
            return None

        wav_files = glob.glob(os.path.join(search_dir, "*.wav"))
        if wav_files:
            return search_dir

        known_subdirs = [
            "read_speech",
            "mnt/dnsv5/clean/read_speech",
            "data/mnt/dnsv5/clean/read_speech",
        ]

        for subdir in known_subdirs:
            check_path = os.path.join(search_dir, subdir)
            if os.path.exists(check_path):
                wav_files = glob.glob(os.path.join(check_path, "*.wav"))
                if wav_files:
                    return check_path

        for root, _dirs, files in os.walk(search_dir):
            wav_files = [f for f in files if f.endswith(".wav")]
            if wav_files:
                return root

        return None

    def _count_wavs_recursive(self, directory: str) -> int:
        if not os.path.exists(directory):
            return 0
        count = 0
        for _root, _dirs, files in os.walk(directory):
            count += len([f for f in files if f.endswith(".wav")])
        return count

    def _collect_wavs_recursive(self, directory: str) -> list[str]:
        wav_files = []
        if not os.path.exists(directory):
            return wav_files
        for root, _dirs, files in os.walk(directory):
            for f in files:
                if f.endswith(".wav"):
                    wav_files.append(os.path.join(root, f))
        return sorted(wav_files)

    def _extract_archive(self, archive_path: str, extract_path: str) -> None:
        logger.info(f"Extracting {os.path.basename(archive_path)}...")

        if not os.path.exists(archive_path):
            raise RuntimeError(f"Archive not found: {archive_path}")

        file_size = os.path.getsize(archive_path)
        file_size_gb = file_size / (1024**3)
        logger.info(f"  Archive size: {file_size_gb:.2f} GB")

        extraction_methods = [
            ["tar", "-xzf", archive_path, "-C", extract_path,
             "--ignore-zeros", "--warning=no-alone-zero-block"],
            ["tar", "-xf", archive_path, "-C", extract_path, "--ignore-zeros"],
        ]

        for i, cmd in enumerate(extraction_methods):
            logger.info(f"  Trying extraction method {i+1}...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            extracted_dir = self._find_extracted_wavs(extract_path)
            if extracted_dir:
                wav_count = self._count_wavs_recursive(extracted_dir)
                if wav_count > 0:
                    logger.info(f"  Extraction successful: {wav_count} WAV files")
                    return

            if result.returncode not in [0, 2]:
                logger.warning(f"  Method {i+1} returned code {result.returncode}")
                if result.stderr:
                    logger.debug(f"  stderr: {result.stderr[:200]}")

        logger.error("All extraction methods failed")
        logger.error(f"Archive size: {file_size_gb:.2f} GB")
        raise RuntimeError(f"Extraction failed: {archive_path}")

    def parse_filename(self, filename: str) -> dict:
        metadata = {
            "book_id": "",
            "chapter": "",
            "reader_id": "",
        }

        basename = os.path.splitext(filename)[0]
        parts = basename.split("_")

        try:
            if len(parts) >= 6:
                if "book" in parts:
                    book_idx = parts.index("book")
                    if book_idx + 1 < len(parts):
                        metadata["book_id"] = parts[book_idx + 1]

                if "chp" in parts:
                    chp_idx = parts.index("chp")
                    if chp_idx + 1 < len(parts):
                        metadata["chapter"] = parts[chp_idx + 1]

                if "reader" in parts:
                    reader_idx = parts.index("reader")
                    if reader_idx + 1 < len(parts):
                        metadata["reader_id"] = parts[reader_idx + 1]
        except (ValueError, IndexError):
            pass

        return metadata

    def collect_audio_files(self, search_dir: str) -> list[dict]:
        entries = []

        if not os.path.exists(search_dir):
            logger.error(f"Directory not found: {search_dir}")
            return entries

        wav_files = self._collect_wavs_recursive(search_dir)
        logger.info(f"Found {len(wav_files)} WAV files in {search_dir}")

        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            metadata = self.parse_filename(filename)

            entry = {
                self.filepath_key: os.path.abspath(wav_path),
                self.text_key: "",
                "sample_rate": SAMPLE_RATE_48KHZ,
                "book_id": metadata.get("book_id", ""),
                "reader_id": metadata.get("reader_id", ""),
            }
            entries.append(entry)

        return entries

    def select_samples(self, entries: list[dict]) -> list[dict]:
        if self.max_samples <= 0:
            logger.info(f"Selected all {len(entries)} samples")
            return entries

        actual_count = min(self.max_samples, len(entries))

        if actual_count < self.max_samples:
            logger.warning(f"Only {actual_count} samples available (requested {self.max_samples})")
        else:
            logger.info(f"Selecting {actual_count} samples")

        return entries[:actual_count]

    def verify_dataset_structure(self, entries: list[dict]) -> None:
        total = len(entries)

        if total == 0:
            logger.error("No audio files found in dataset!")
            return

        logger.info("=" * 60)
        logger.info("Dataset Structure Verification")
        logger.info("=" * 60)
        logger.info(f"Total samples: {total}")

        unique_readers = set()
        unique_books = set()
        for entry in entries:
            if entry.get("reader_id"):
                unique_readers.add(entry["reader_id"])
            if entry.get("book_id"):
                unique_books.add(entry["book_id"])

        logger.info(f"Unique readers: {len(unique_readers)}")
        logger.info(f"Unique books: {len(unique_books)}")

        samples_to_check = min(5, total)
        missing_files = []
        for entry in entries[:samples_to_check]:
            filepath = entry.get(self.filepath_key, "")
            if not os.path.exists(filepath):
                missing_files.append(filepath)

        if missing_files:
            logger.warning(f"Missing files detected: {missing_files[:3]}...")
        else:
            logger.info(f"File existence verified for {samples_to_check} samples")

        if entries:
            logger.info(f"Sample entry: {entries[0]}")

        logger.info("=" * 60)

    def process(self, _: _EmptyTask) -> list[AudioTask]:
        """
        Main processing method. Returns list[AudioTask] with one AudioTask per file.
        """
        if self.auto_download:
            logger.info("Auto-download enabled. Downloading dataset...")
            search_dir = self.download_and_extract()
        else:
            search_dir = self._find_extracted_wavs(self.raw_data_dir)
            if not search_dir:
                search_dir = self.raw_data_dir
                logger.warning(f"No WAV files found, searching in: {search_dir}")

        entries = self.collect_audio_files(search_dir)
        self.verify_dataset_structure(entries)

        if not entries:
            logger.error("No audio files found in the dataset")
            return []

        selected_entries = self.select_samples(entries)
        logger.info(f"Creating manifest with {len(selected_entries)} total samples")

        audio_tasks = []
        for i, entry in enumerate(selected_entries):
            audio_tasks.append(AudioTask(
                data=entry,
                task_id=f"readspeech_{i}",
                dataset_name="DNS-ReadSpeech",
                filepath_key=self.filepath_key,
            ))

        return audio_tasks
