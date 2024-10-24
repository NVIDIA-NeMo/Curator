# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from setuptools import setup, find_packages
import pathlib
from itertools import chain

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")


def strtobool(value: str) -> bool:
    value = value.lower()
    if value in ("y", "yes", "1", "true"):
        return True
    return False


def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename), encoding="utf-8") as f:
        content = f.readlines()
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")

cuda12_requirements_filename = (
    "requirements_rapids_nightly.txt"
    if strtobool(os.getenv("RAPIDS_NIGHTLY", "false"))
    else "requirements_cuda12x.txt"
)

extras_require: dict = {
    "cuda12x": req_file(cuda12_requirements_filename),
    "image": req_file("requirements_image.txt"),
}

extras_require["all"] = list(chain(extras_require.values()))

extras_require["image"] = list(
    chain([extras_require["image"], extras_require["cuda12x"]])
)

setup(
    name="nemo_curator",
    version="0.5.0",
    description="Scalable Data Preprocessing Tool for "
    "Training Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/NeMo-Curator",
    author="Joseph Jennings, Mostofa Patwary, Sandeep Subramanian, "
    "Shrimai Prabhumoye, Ayush Dattagupta, Vibhu Jawa, Jiwei Liu, Ryan Wolf",
    author_email="jjennings@nvidia.com, mpatwary@nvidia.com, "
    "rywolf@nvidia.com, sprabhumoye@nvidia.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    python_requires=">=3.10, <3.11",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "get_common_crawl_urls=nemo_curator.scripts.get_common_crawl_urls:console_script",
            "get_wikipedia_urls=nemo_curator.scripts.get_wikipedia_urls:console_script",
            "download_and_extract=nemo_curator.scripts.download_and_extract:console_script",
            "text_cleaning=nemo_curator.scripts.text_cleaning:console_script",
            "add_id=nemo_curator.scripts.add_id:console_script",
            "get_metadata_from_corpus=nemo_curator.get_metadata_from_corpus:console_script",
            "make_data_shards=nemo_curator.scripts.make_data_shards:console_script",
            "prepare_fasttext_training_data=nemo_curator.scripts.prepare_fasttext_training_data:console_script",
            "train_fasttext=nemo_curator.scripts.train_fasttext:console_script",
            "filter_documents=nemo_curator.scripts.filter_documents:console_script",
            "separate_by_metadata=nemo_curator.scripts.separate_by_metadata:console_script",
            "prepare_task_data=nemo_curator.scripts.prepare_task_data:console_script",
            "find_matching_ngrams=nemo_curator.scripts.find_matching_ngrams:console_script",
            "remove_matching_ngrams=nemo_curator.scripts.remove_matching_ngrams:console_script",
            "gpu_compute_minhashes=nemo_curator.scripts.fuzzy_deduplication.compute_minhashes:console_script",
            "minhash_buckets=nemo_curator.scripts.fuzzy_deduplication.minhash_lsh:console_script",
            "jaccard_map_buckets=nemo_curator.scripts.fuzzy_deduplication.map_buckets:console_script",
            "jaccard_shuffle=nemo_curator.scripts.fuzzy_deduplication.jaccard_shuffle:console_script",
            "jaccard_compute=nemo_curator.scripts.fuzzy_deduplication.jaccard_compute:console_script",
            "gpu_connected_component=nemo_curator.scripts.fuzzy_deduplication.connected_components:console_script",
            "gpu_exact_dups=nemo_curator.scripts.find_exact_duplicates:console_script",
            "deidentify=nemo_curator.scripts.find_pii_and_deidentify:console_script",
            "domain_classifier_inference=nemo_curator.scripts.classifiers.domain_classifier_inference:console_script",
            "quality_classifier_inference=nemo_curator.scripts.classifiers.quality_classifier_inference:console_script",
            "aegis_classifier_inference=nemo_curator.scripts.classifiers.aegis_classifier_inference:console_script",
            "verify_classification_results=nemo_curator.scripts.verify_classification_results:console_script",
            "blend_datasets=nemo_curator.scripts.blend_datasets:console_script",
            "semdedup_extract_embeddings=nemo_curator.scripts.semdedup.compute_embeddings:console_script",
            "semdedup_clustering=nemo_curator.scripts.semdedup.clustering:console_script",
            "semdedup_extract_dedup_ids=nemo_curator.scripts.semdedup.extract_dedup_data:console_script",
        ],
    },
)
