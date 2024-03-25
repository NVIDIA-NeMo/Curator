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

from pydoc import locate

import yaml

import nemo_curator
from nemo_curator.download.doc_builder import (
    import_downloader,
    import_extractor,
    import_iterator,
)
from nemo_curator.filters import import_filter
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir


def build_filter(filter_config):
    # Import the filter
    filter_class = import_filter(filter_config["name"])

    # Check if constructor has been provided
    if ("params" not in filter_config) or (filter_config["params"] is None):
        filter_config["params"] = {}

    doc_filter = filter_class(**filter_config["params"])

    if filter_config.get("filter_only", False):
        filter_stage = nemo_curator.Filter(
            doc_filter.keep_document, filter_field=doc_filter.name
        )
    else:
        score_field = (
            doc_filter._name if filter_config.get("log_score", False) else None
        )
        filter_stage = nemo_curator.ScoreFilter(
            doc_filter, filter_config.get("input_field"), score_field=score_field
        )

    return filter_stage


def build_filter_pipeline(filter_config_file):
    # Get the filter config file
    with open(filter_config_file, "r") as config_file:
        filter_params = yaml.load(config_file, Loader=yaml.FullLoader)

    filters = []
    text_field = filter_params.get("input_field")
    for nc_filter_config in filter_params.get("filters"):
        if (
            "input_field" not in nc_filter_config
            or nc_filter_config["input_field"] is None
        ):
            nc_filter_config["input_field"] = text_field
        new_filter = build_filter(nc_filter_config)
        filters.append(new_filter)

    return nemo_curator.Sequential(filters)


def build_downloader(downloader_config_file, default_download_dir=None):
    # Get the downloader config file
    with open(downloader_config_file, "r") as config_file:
        downloader_params = yaml.load(config_file, Loader=yaml.FullLoader)

    download_class = import_downloader(downloader_params["download_module"])
    no_download_dir = ("download_dir" not in downloader_params["download_params"]) or (
        downloader_params["download_params"] is None
    )
    if no_download_dir and default_download_dir:
        downloader_params["download_params"]["download_dir"] = default_download_dir
    expand_outdir_and_mkdir(downloader_params["download_params"]["download_dir"])
    downloader = download_class(**downloader_params["download_params"])

    iterator_class = import_iterator(downloader_params["iterator_module"])
    iterator = iterator_class(**downloader_params["iterator_params"])

    extractor_class = import_extractor(downloader_params["extract_module"])
    extractor = extractor_class(**downloader_params["extract_params"])

    dataset_format = {}
    for field, field_type in downloader_params["format"].items():
        dataset_format[field] = locate(field_type)

    return downloader, iterator, extractor, dataset_format
