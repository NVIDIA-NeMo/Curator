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

import glob
import os
import subprocess
from dataclasses import dataclass

from loguru import logger

from nemo_curator.stages.audio.datasets.file_utils import download_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch, _EmptyTask

# Sample rate constant (DNS Challenge read_speech is 48kHz)
SAMPLE_RATE_48KHZ = 48000

# DNS Challenge 5 Azure URLs
DNS_CHALLENGE_BASE_URL = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"

# Available parts for download (these are SPLIT archives - must be concatenated before extraction)
DNS_READSPEECH_PARTS = [
    "Track1_Headset/read_speech.tgz.partaa",
    "Track1_Headset/read_speech.tgz.partab",
    "Track1_Headset/read_speech.tgz.partac",
    "Track1_Headset/read_speech.tgz.partad",
    "Track1_Headset/read_speech.tgz.partae",
    "Track1_Headset/read_speech.tgz.partaf",
]


@dataclass
class CreateInitialManifestReadSpeechStage(ProcessingStage[_EmptyTask, AudioBatch]):
    """
    Stage to create initial manifest for the DNS Challenge Read Speech dataset.

    Dataset: Microsoft DNS Challenge 5 - Read Speech (Track 1 Headset)
    Source: https://github.com/microsoft/DNS-Challenge

    **Auto-Download Support**: This stage can automatically download and extract the
    DNS Challenge Read Speech dataset when `auto_download=True`.

    **IMPORTANT**: The DNS Challenge archives are SPLIT files (partaa, partab, etc.).
    These parts must be concatenated together before extraction - they cannot be
    extracted independently. Only `partaa` (part 1) can be extracted alone as it
    contains the beginning of the archive.

    **Directory Structure After Download**:
    ```
    raw_data_dir/
    └── read_speech/
        ├── book_00000_chp_0001_reader_xxxxx_*.wav
        └── ... (all extracted WAV files)
    ```

    Args:
        raw_data_dir (str): Directory where data will be downloaded/extracted to.
        max_samples (int): Maximum number of samples to include. Default is 5000.
            Set to -1 for all available samples.
        auto_download (bool): If True, automatically download and extract dataset.
            If False, expects data to already exist. Default is True.
        download_parts (int): Number of parts to download (1-6). Default is 1.
            Note: Parts 2-6 require part 1 as they are split archive continuations.

    Returns:
        AudioBatch objects with:
            {
                "audio_filepath": <absolute path to WAV file>,
                "text": "",  # No transcription available for DNS data
                "sample_rate": 48000,
                "book_id": <extracted book ID>,
                "reader_id": <extracted reader ID>,
            }
    """

    raw_data_dir: str
    max_samples: int = 5000
    auto_download: bool = True
    download_parts: int = 1  # 1-6 parts
    filepath_key: str = "audio_filepath"
    text_key: str = "text"
    name: str = "CreateInitialManifestReadSpeech"
    batch_size: int = 1

    def download_and_extract(self) -> str:
        """
        Download and extract DNS Challenge Read Speech dataset.
        
        Split archives (partaa, partab, etc.) are concatenated before extraction.
        All WAV files are extracted to a single read_speech/ directory.

        Returns:
            Path to the extracted read_speech directory containing WAV files
        """
        if self.download_parts < 1 or self.download_parts > 6:
            msg = f"download_parts must be between 1 and 6, got {self.download_parts}"
            raise ValueError(msg)

        # Create output directory
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # Check if already extracted
        extract_dir = os.path.join(self.raw_data_dir, "read_speech")
        existing_dir = self._find_extracted_wavs(self.raw_data_dir)
        if existing_dir:
            wav_count = self._count_wavs_recursive(existing_dir)
            logger.info(f"Dataset already extracted at {existing_dir} ({wav_count} WAV files)")
            return existing_dir

        logger.info("=" * 60)
        logger.info("DNS Challenge 5 - Read Speech Download")
        logger.info("=" * 60)
        logger.info(f"Downloading {self.download_parts} part(s) to: {self.raw_data_dir}")
        
        if self.download_parts == 1:
            logger.info("Single part download - will extract directly")
        else:
            logger.info(f"Multi-part download - will concatenate {self.download_parts} parts before extraction")
            logger.info("NOTE: Split archives must be combined - parts 2+ cannot be extracted alone")
        
        logger.info("=" * 60)

        # Download all parts
        downloaded_files = []
        for part_num in range(1, self.download_parts + 1):
            part_index = part_num - 1
            part_name = DNS_READSPEECH_PARTS[part_index]
            url = f"{DNS_CHALLENGE_BASE_URL}/{part_name}"
            filename = os.path.basename(part_name)
            filepath = os.path.join(self.raw_data_dir, filename)

            logger.info(f"\nDownloading part {part_num}/{self.download_parts}: {filename}")

            # Check if file already downloaded
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                if file_size > 0:
                    logger.info(f"  Already exists: {file_size / (1024**3):.2f} GB")
                    downloaded_files.append(filepath)
                    continue
                else:
                    logger.warning(f"  Removing empty file")
                    os.remove(filepath)

            # Download
            filepath = download_file(url, self.raw_data_dir, verbose=True)
            
            # Verify
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                logger.error(f"Downloaded file is empty: {filepath}")
                os.remove(filepath)
                raise RuntimeError(f"Download failed - empty file: {filename}")

            logger.info(f"  ✓ Downloaded: {file_size / (1024**3):.2f} GB")
            downloaded_files.append(filepath)

        logger.info("\n" + "=" * 60)
        logger.info("All parts downloaded. Preparing extraction...")
        logger.info("=" * 60)

        # Prepare archive for extraction
        if self.download_parts == 1:
            # Single part - extract directly with --ignore-zeros for partial archive
            archive_to_extract = downloaded_files[0]
            logger.info(f"Extracting single part: {os.path.basename(archive_to_extract)}")
        else:
            # Multiple parts - concatenate first
            combined_archive = os.path.join(self.raw_data_dir, "read_speech_combined.tgz")
            
            if os.path.exists(combined_archive):
                logger.info(f"Combined archive already exists: {combined_archive}")
            else:
                logger.info(f"Concatenating {len(downloaded_files)} parts...")
                
                # Sort files to ensure correct order (partaa, partab, partac, ...)
                downloaded_files.sort()
                
                with open(combined_archive, 'wb') as outfile:
                    for i, part_file in enumerate(downloaded_files):
                        logger.info(f"  Adding part {i+1}: {os.path.basename(part_file)}")
                        with open(part_file, 'rb') as infile:
                            # Read and write in chunks to handle large files
                            while True:
                                chunk = infile.read(64 * 1024 * 1024)  # 64MB chunks
                                if not chunk:
                                    break
                                outfile.write(chunk)
                
                combined_size = os.path.getsize(combined_archive)
                logger.info(f"  ✓ Combined archive: {combined_size / (1024**3):.2f} GB")
            
            archive_to_extract = combined_archive

        # Extract
        logger.info(f"\nExtracting archive...")
        self._extract_archive(archive_to_extract, self.raw_data_dir)

        # Find extracted files
        extracted_dir = self._find_extracted_wavs(self.raw_data_dir)
        if not extracted_dir:
            raise RuntimeError("Extraction failed - no WAV files found")

        wav_count = self._count_wavs_recursive(extracted_dir)
        logger.info(f"\n✓ Extraction complete: {wav_count} WAV files in {extracted_dir}")

        # Clean up downloaded archives
        logger.info("\nCleaning up archives...")
        for part_file in downloaded_files:
            if os.path.exists(part_file):
                os.remove(part_file)
                logger.info(f"  Removed: {os.path.basename(part_file)}")
        
        if self.download_parts > 1:
            combined_archive = os.path.join(self.raw_data_dir, "read_speech_combined.tgz")
            if os.path.exists(combined_archive):
                os.remove(combined_archive)
                logger.info(f"  Removed: read_speech_combined.tgz")

        logger.info("=" * 60)
        logger.info(f"✓ Dataset ready: {wav_count} WAV files")
        logger.info(f"  Location: {extracted_dir}")
        logger.info("=" * 60)

        return extracted_dir

    def _find_extracted_wavs(self, search_dir: str) -> str | None:
        """
        Recursively search for directory containing WAV files.
        
        Args:
            search_dir: Directory to search in
            
        Returns:
            Path to directory containing WAV files, or None if not found
        """
        if not os.path.exists(search_dir):
            return None

        # Check current directory
        wav_files = glob.glob(os.path.join(search_dir, "*.wav"))
        if wav_files:
            return search_dir

        # Known extraction paths from DNS archive
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

        # Recursive search as fallback
        for root, dirs, files in os.walk(search_dir):
            wav_files = [f for f in files if f.endswith(".wav")]
            if wav_files:
                return root

        return None

    def _count_wavs_recursive(self, directory: str) -> int:
        """Count WAV files recursively in a directory."""
        if not os.path.exists(directory):
            return 0
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len([f for f in files if f.endswith(".wav")])
        return count

    def _collect_wavs_recursive(self, directory: str) -> list[str]:
        """Collect all WAV file paths recursively from a directory."""
        wav_files = []
        if not os.path.exists(directory):
            return wav_files
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith(".wav"):
                    wav_files.append(os.path.join(root, f))
        return sorted(wav_files)

    def _extract_archive(self, archive_path: str, extract_path: str) -> None:
        """
        Extract a tar.gz archive using tar command.
        
        Handles split/partial archives using --ignore-zeros flag.
        """
        logger.info(f"Extracting {os.path.basename(archive_path)}...")

        if not os.path.exists(archive_path):
            raise RuntimeError(f"Archive not found: {archive_path}")

        file_size = os.path.getsize(archive_path)
        file_size_gb = file_size / (1024**3)
        logger.info(f"  Archive size: {file_size_gb:.2f} GB")

        # Try extraction methods in order
        extraction_methods = [
            # Method 1: tar with gzip and ignore-zeros
            ["tar", "-xzf", archive_path, "-C", extract_path, 
             "--ignore-zeros", "--warning=no-alone-zero-block"],
            # Method 2: plain tar (no gzip)
            ["tar", "-xf", archive_path, "-C", extract_path, "--ignore-zeros"],
        ]

        for i, cmd in enumerate(extraction_methods):
            logger.info(f"  Trying extraction method {i+1}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check if any WAV files were extracted
            extracted_dir = self._find_extracted_wavs(extract_path)
            if extracted_dir:
                wav_count = self._count_wavs_recursive(extracted_dir)
                if wav_count > 0:
                    logger.info(f"  ✓ Extraction successful: {wav_count} WAV files")
                    return

            if result.returncode not in [0, 2]:  # 2 is just warnings
                logger.warning(f"  Method {i+1} returned code {result.returncode}")
                if result.stderr:
                    logger.debug(f"  stderr: {result.stderr[:200]}")

        # All methods failed
        logger.error("All extraction methods failed")
        logger.error(f"Archive size: {file_size_gb:.2f} GB")
        raise RuntimeError(f"Extraction failed: {archive_path}")

    def parse_filename(self, filename: str) -> dict:
        """
        Parse the DNS read_speech filename to extract metadata.

        Format: book_XXXXX_chp_XXXX_reader_XXXXX_X_seg_X_segX.wav

        Returns:
            Dictionary with extracted metadata (book_id, chapter, reader_id, etc.)
        """
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
        """
        Collect audio files from the extracted directory.

        Args:
            search_dir: Directory containing WAV files

        Returns:
            List of entry dictionaries with audio_filepath and metadata
        """
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
        """
        Select samples based on max_samples configuration.
        """
        if self.max_samples <= 0:
            logger.info(f"Selected all {len(entries)} samples")
            return entries

        actual_count = min(self.max_samples, len(entries))

        if actual_count < self.max_samples:
            logger.warning(f"Only {actual_count} samples available (requested {self.max_samples})")
        else:
            logger.info(f"Selecting {actual_count} samples")

        return entries[:actual_count]

    def create_batches(self, entries: list[dict]) -> list[AudioBatch]:
        """
        Create AudioBatch objects from entries.
        """
        speech_tasks = []
        batch_entries = []

        for i, entry in enumerate(entries):
            batch_entries.append(entry)

            if len(batch_entries) == self.batch_size:
                speech_task = AudioBatch(
                    task_id=f"readspeech_batch_{i // self.batch_size}",
                    dataset_name="DNS-ReadSpeech",
                    filepath_key=self.filepath_key,
                    data=batch_entries if self.batch_size > 1 else batch_entries[0],
                )
                speech_tasks.append(speech_task)
                batch_entries = []

        # Handle remaining entries
        if batch_entries:
            speech_task = AudioBatch(
                task_id=f"readspeech_batch_{len(entries) // self.batch_size}",
                dataset_name="DNS-ReadSpeech",
                filepath_key=self.filepath_key,
                data=batch_entries if len(batch_entries) > 1 else batch_entries[0],
            )
            speech_tasks.append(speech_task)

        return speech_tasks

    def verify_dataset_structure(self, entries: list[dict]) -> None:
        """
        Verify that the dataset has the expected structure.
        """
        total = len(entries)

        if total == 0:
            logger.error("No audio files found in dataset!")
            return

        logger.info("=" * 60)
        logger.info("Dataset Structure Verification")
        logger.info("=" * 60)
        logger.info(f"Total samples: {total}")

        # Count unique readers and books
        unique_readers = set()
        unique_books = set()
        for entry in entries:
            if entry.get("reader_id"):
                unique_readers.add(entry["reader_id"])
            if entry.get("book_id"):
                unique_books.add(entry["book_id"])

        logger.info(f"Unique readers: {len(unique_readers)}")
        logger.info(f"Unique books: {len(unique_books)}")

        # Verify file existence
        samples_to_check = min(5, total)
        missing_files = []
        for entry in entries[:samples_to_check]:
            filepath = entry.get(self.filepath_key, "")
            if not os.path.exists(filepath):
                missing_files.append(filepath)

        if missing_files:
            logger.warning(f"Missing files detected: {missing_files[:3]}...")
        else:
            logger.info(f"File existence verified for {samples_to_check} samples ✓")

        if entries:
            logger.info(f"Sample entry: {entries[0]}")

        logger.info("=" * 60)

    def process(self, _: _EmptyTask) -> list[AudioBatch]:
        """
        Main processing method.

        Processing Steps:
        1. Downloads all requested parts
        2. Concatenates parts if multiple (split archive format)
        3. Extracts the archive to read_speech/ directory
        4. Collects all WAV files and creates AudioBatch tasks

        Returns:
            List of AudioBatch tasks
        """
        if self.auto_download:
            logger.info("Auto-download enabled. Downloading dataset...")
            search_dir = self.download_and_extract()
        else:
            # Find existing data
            search_dir = self._find_extracted_wavs(self.raw_data_dir)
            if not search_dir:
                search_dir = self.raw_data_dir
                logger.warning(f"No WAV files found, searching in: {search_dir}")

        # Collect audio files
        entries = self.collect_audio_files(search_dir)

        # Verify structure
        self.verify_dataset_structure(entries)

        if not entries:
            logger.error("No audio files found in the dataset")
            return []

        # Select samples
        selected_entries = self.select_samples(entries)

        logger.info(f"Creating manifest with {len(selected_entries)} total samples")

        # Create batches
        return self.create_batches(selected_entries)
