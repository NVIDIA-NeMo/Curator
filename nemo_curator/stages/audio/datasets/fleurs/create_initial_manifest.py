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

import os
from dataclasses import dataclass

from nemo_curator.stages.audio.datasets.file_utils import download_file, extract_archive
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch, _EmptyTask


def get_fleurs_url_list(lang: str, split: str) -> list[str]:
    """
    examples
    "https://huggingface.co/datasets/google/fleurs/resolve/main/data/hy_am/audio/dev.tar.gz",
    "https://huggingface.co/datasets/google/fleurs/resolve/main/data/hy_am/dev.tsv"

    """

    urls = []
    base_url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"

    base_lang_url = os.path.join(base_url, lang)
    tsv_url = f"{base_lang_url}/{split}.tsv"
    urls.append(tsv_url)

    tar_gz_url = f"{base_lang_url}/audio/{split}.tar.gz"
    urls.append(tar_gz_url)

    return urls


@dataclass
class CreateInitialManifestFleursStage(ProcessingStage[_EmptyTask, AudioBatch]):
    """
    Stage to create initial manifest for the FLEURS dataset.

    Dataset link: https://huggingface.co/datasets/google/fleurs

    Will download all files, extract them, and create a manifest file with the
    "audio_filepath" and "text" fields.

    Args:
        lang (str): Language to be processed, identified by a combination of ISO 639-1 and ISO 3166-1 alpha-2 codes.
            Examples are:

            - ``"hy_am"`` for Armenian
            - ``"ko_kr"`` for Korean

        split (str): Which dataset splits to process.
            Options are:

            - ``"test"``
            - ``"train"``
            - ``"dev"``

        raw_data_dir (str): Path to the folder where the data archive should be downloaded and extracted.

    Returns:
        This srage generates an initial SpeechObject with the following fields:

            {
                "audio_filepath": <path to the audio file>,
                "text": <transcription>,
            }
    """

    lang: str
    split: str
    raw_data_dir: str
    filepath_key: str = "audio_filepath"
    text_key: str = "text"
    _name: str = "CreateInitialManifestFleurs"
    _batch_size: int = 1

    def process_transcript(self, file_path: str) -> list[AudioBatch]:
        """
        Parse transcript TSV file and put it inside manifest.
        Assumes the TSV file has two columns: file name and text.
        """

        speech_tasks = []
        root = os.path.splitext(file_path)[0]
        min_num_parts = 2  # Skip lines that don't have at least 2 parts
        entries = []
        count = 0
        with open(file_path, encoding="utf-8") as fin:
            for line in fin:
                # Split the line into filename text using the tab delimiter
                parts = line.strip().split("\t")
                if len(parts) < min_num_parts:
                    continue

                file_name, transcript_text = parts[1], parts[2]
                wav_file = os.path.join(root, file_name)

                entries.append({self.filepath_key: os.path.abspath(wav_file), self.text_key: transcript_text})
                count += 1
                if count == self.batch_size:
                    speech_task = AudioBatch(
                        task_id=f"task_id_{file_path}",
                        dataset_name=f"Fleurs_{self.lang}_{self.split}_{self.raw_data_dir}",
                        filepath_key=self.filepath_key,
                        data=entries,
                    )
                    entries = []
                    count = 0
                    speech_tasks.append(speech_task)
            if count > 0:
                speech_task = AudioBatch(
                    task_id=f"task_id_{file_path}",
                    dataset_name=f"Fleurs_{self.lang}_{self.split}_{self.raw_data_dir}",
                    filepath_key=self.filepath_key,
                    data=entries,
                )
                speech_tasks.append(speech_task)
        return speech_tasks

    def download_extract_files(self, dst_folder: str) -> None:
        """downloading and extracting files"""

        os.makedirs(dst_folder, exist_ok=True)

        # downloading all files
        for file_url in get_fleurs_url_list(self.lang, self.split):
            download_file(file_url, str(dst_folder))

        extract_archive(f"{dst_folder}/{self.split}.tar.gz", str(dst_folder), force_extract=True)

    def process(self, _: _EmptyTask) -> list[AudioBatch]:
        self.download_extract_files(self.raw_data_dir)
        return self.process_transcript(os.path.join(self.raw_data_dir, self.split + ".tsv"))
