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
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

import nemo_curator
from nemo_curator.utils.config_utils import (
    build_downloader,
    build_filter,
    build_filter_pipeline,
)


class TestBuildFilter:
    """Tests for the build_filter function."""

    def test_build_filter_with_params(self):
        """Test building a filter with parameters."""
        # Mock the imported filter class
        mock_filter_class = MagicMock()
        mock_filter_instance = MagicMock()
        mock_filter_class.return_value = mock_filter_instance
        mock_filter_instance.name = "test_filter"
        mock_filter_instance._name = "test_filter"

        # Create filter config
        filter_config = {
            "name": "nemo_curator.filters.test_filter",
            "params": {"param1": "value1", "param2": "value2"},
            "input_field": "text",
        }

        # Mock the import_filter function
        with patch(
            "nemo_curator.utils.config_utils.import_filter",
            return_value=mock_filter_class,
        ):
            # Mock the ScoreFilter class
            with patch("nemo_curator.ScoreFilter") as mock_score_filter:
                mock_score_filter_instance = MagicMock()
                mock_score_filter.return_value = mock_score_filter_instance

                # Call the function
                result = build_filter(filter_config)

                # Check assertions
                mock_filter_class.assert_called_once_with(
                    param1="value1", param2="value2"
                )
                mock_score_filter.assert_called_once_with(
                    mock_filter_instance, filter_config["input_field"], score_field=None
                )
                assert result == mock_score_filter_instance

    def test_build_filter_without_params(self):
        """Test building a filter without parameters."""
        # Mock the imported filter class
        mock_filter_class = MagicMock()
        mock_filter_instance = MagicMock()
        mock_filter_class.return_value = mock_filter_instance
        mock_filter_instance.name = "test_filter"
        mock_filter_instance._name = "test_filter"

        # Create filter config without params
        filter_config = {
            "name": "nemo_curator.filters.test_filter",
            "input_field": "text",
        }

        # Mock the import_filter function
        with patch(
            "nemo_curator.utils.config_utils.import_filter",
            return_value=mock_filter_class,
        ):
            # Mock the ScoreFilter class
            with patch("nemo_curator.ScoreFilter") as mock_score_filter:
                mock_score_filter_instance = MagicMock()
                mock_score_filter.return_value = mock_score_filter_instance

                # Call the function
                result = build_filter(filter_config)

                # Check assertions
                mock_filter_class.assert_called_once_with()
                mock_score_filter.assert_called_once_with(
                    mock_filter_instance, filter_config["input_field"], score_field=None
                )
                assert result == mock_score_filter_instance

    def test_build_filter_with_none_params(self):
        """Test building a filter with params explicitly set to None."""
        # Mock the imported filter class
        mock_filter_class = MagicMock()
        mock_filter_instance = MagicMock()
        mock_filter_class.return_value = mock_filter_instance
        mock_filter_instance.name = "test_filter"
        mock_filter_instance._name = "test_filter"

        # Create filter config with params=None
        filter_config = {
            "name": "nemo_curator.filters.test_filter",
            "params": None,
            "input_field": "text",
        }

        # Mock the import_filter function
        with patch(
            "nemo_curator.utils.config_utils.import_filter",
            return_value=mock_filter_class,
        ):
            # Mock the ScoreFilter class
            with patch("nemo_curator.ScoreFilter") as mock_score_filter:
                mock_score_filter_instance = MagicMock()
                mock_score_filter.return_value = mock_score_filter_instance

                # Call the function
                result = build_filter(filter_config)

                # Check assertions
                mock_filter_class.assert_called_once_with()
                mock_score_filter.assert_called_once_with(
                    mock_filter_instance, filter_config["input_field"], score_field=None
                )
                assert result == mock_score_filter_instance

    def test_build_filter_with_log_score(self):
        """Test building a filter with log_score=True."""
        # Mock the imported filter class
        mock_filter_class = MagicMock()
        mock_filter_instance = MagicMock()
        mock_filter_class.return_value = mock_filter_instance
        mock_filter_instance.name = "test_filter"
        mock_filter_instance._name = "test_filter"

        # Create filter config with log_score
        filter_config = {
            "name": "nemo_curator.filters.test_filter",
            "params": {"param1": "value1"},
            "input_field": "text",
            "log_score": True,
        }

        # Mock the import_filter function
        with patch(
            "nemo_curator.utils.config_utils.import_filter",
            return_value=mock_filter_class,
        ):
            # Mock the ScoreFilter class
            with patch("nemo_curator.ScoreFilter") as mock_score_filter:
                mock_score_filter_instance = MagicMock()
                mock_score_filter.return_value = mock_score_filter_instance

                # Call the function
                result = build_filter(filter_config)

                # Check assertions
                mock_filter_class.assert_called_once_with(param1="value1")
                mock_score_filter.assert_called_once_with(
                    mock_filter_instance,
                    filter_config["input_field"],
                    score_field="test_filter",
                )
                assert result == mock_score_filter_instance

    def test_build_filter_filter_only(self):
        """Test building a filter with filter_only=True."""
        # Mock the imported filter class
        mock_filter_class = MagicMock()
        mock_filter_instance = MagicMock()
        mock_filter_class.return_value = mock_filter_instance
        mock_filter_instance.name = "test_filter"
        mock_filter_instance.keep_document = MagicMock()

        # Create filter config with filter_only
        filter_config = {
            "name": "nemo_curator.filters.test_filter",
            "params": {"param1": "value1"},
            "filter_only": True,
        }

        # Mock the import_filter function
        with patch(
            "nemo_curator.utils.config_utils.import_filter",
            return_value=mock_filter_class,
        ):
            # Mock the Filter class
            with patch("nemo_curator.Filter") as mock_filter:
                mock_filter_instance_outer = MagicMock()
                mock_filter.return_value = mock_filter_instance_outer

                # Call the function
                result = build_filter(filter_config)

                # Check assertions
                mock_filter_class.assert_called_once_with(param1="value1")
                mock_filter.assert_called_once_with(
                    mock_filter_instance.keep_document,
                    filter_field=mock_filter_instance.name,
                )
                assert result == mock_filter_instance_outer


class TestBuildFilterPipeline:
    """Tests for the build_filter_pipeline function."""

    def test_build_filter_pipeline(self):
        """Test building a filter pipeline from a config file."""
        # Create mock filter config
        filter_params = {
            "input_field": "text",
            "filters": [
                {
                    "name": "nemo_curator.filters.filter1",
                    "params": {"param1": "value1"},
                },
                {
                    "name": "nemo_curator.filters.filter2",
                    "params": {"param2": "value2"},
                    "input_field": "custom_field",
                },
                {
                    "name": "nemo_curator.filters.filter3",
                    "params": {"param3": "value3"},
                    "input_field": None,
                },
            ],
        }

        # Mock the open function and yaml.load
        mock_yaml_load = yaml.load
        with patch("builtins.open", mock_open(read_data="dummy")) as mock_file:
            with patch("yaml.load", return_value=filter_params) as mock_load:
                # Mock the build_filter function
                mock_filter1 = MagicMock()
                mock_filter2 = MagicMock()
                mock_filter3 = MagicMock()
                with patch(
                    "nemo_curator.utils.config_utils.build_filter",
                    side_effect=[mock_filter1, mock_filter2, mock_filter3],
                ) as mock_build_filter:
                    # Mock the Sequential class
                    with patch("nemo_curator.Sequential") as mock_sequential:
                        mock_sequential_instance = MagicMock()
                        mock_sequential.return_value = mock_sequential_instance

                        # Call the function
                        result = build_filter_pipeline("dummy_path.yaml")

                        # Check assertions
                        mock_file.assert_called_once_with("dummy_path.yaml", "r")
                        mock_load.assert_called_once()
                        assert mock_build_filter.call_count == 3
                        # First filter should use the input_field from the config
                        assert mock_build_filter.call_args_list[0][0][0] == {
                            "name": "nemo_curator.filters.filter1",
                            "params": {"param1": "value1"},
                            "input_field": "text",
                        }
                        # Second filter should use its own input_field
                        assert mock_build_filter.call_args_list[1][0][0] == {
                            "name": "nemo_curator.filters.filter2",
                            "params": {"param2": "value2"},
                            "input_field": "custom_field",
                        }
                        # Third filter should have input_field replaced with parent's input_field
                        assert mock_build_filter.call_args_list[2][0][0] == {
                            "name": "nemo_curator.filters.filter3",
                            "params": {"param3": "value3"},
                            "input_field": "text",
                        }
                        mock_sequential.assert_called_once_with(
                            [mock_filter1, mock_filter2, mock_filter3]
                        )
                        assert result == mock_sequential_instance


class TestBuildDownloader:
    """Tests for the build_downloader function."""

    def test_build_downloader_with_download_dir(self):
        """Test building a downloader with a specified download directory."""
        # Create mock downloader config
        downloader_params = {
            "download_module": "nemo_curator.download.test_downloader",
            "download_params": {
                "download_dir": "/path/to/download",
                "param1": "value1",
            },
            "iterator_module": "nemo_curator.download.test_iterator",
            "iterator_params": {"param2": "value2"},
            "extract_module": "nemo_curator.download.test_extractor",
            "extract_params": {"param3": "value3"},
            "format": {"field1": "str", "field2": "int"},
        }

        # Mock the open function and yaml.load
        with patch("builtins.open", mock_open(read_data="dummy")) as mock_file:
            with patch("yaml.load", return_value=downloader_params) as mock_load:
                # Mock the import functions and classes
                mock_downloader_class = MagicMock()
                mock_downloader_instance = MagicMock()
                mock_downloader_class.return_value = mock_downloader_instance

                mock_iterator_class = MagicMock()
                mock_iterator_instance = MagicMock()
                mock_iterator_class.return_value = mock_iterator_instance

                mock_extractor_class = MagicMock()
                mock_extractor_instance = MagicMock()
                mock_extractor_class.return_value = mock_extractor_instance

                # Mock locate for format types
                def mock_locate(type_name):
                    if type_name == "str":
                        return str
                    elif type_name == "int":
                        return int
                    return None

                with patch(
                    "nemo_curator.utils.config_utils.import_downloader",
                    return_value=mock_downloader_class,
                ) as mock_import_downloader:
                    with patch(
                        "nemo_curator.utils.config_utils.import_iterator",
                        return_value=mock_iterator_class,
                    ) as mock_import_iterator:
                        with patch(
                            "nemo_curator.utils.config_utils.import_extractor",
                            return_value=mock_extractor_class,
                        ) as mock_import_extractor:
                            with patch(
                                "nemo_curator.utils.config_utils.locate",
                                side_effect=mock_locate,
                            ) as mock_locate_func:
                                with patch(
                                    "nemo_curator.utils.config_utils.expand_outdir_and_mkdir"
                                ) as mock_expand:
                                    # Call the function
                                    downloader, iterator, extractor, dataset_format = (
                                        build_downloader("dummy_path.yaml")
                                    )

                                    # Check assertions
                                    mock_file.assert_called_once_with(
                                        "dummy_path.yaml", "r"
                                    )
                                    mock_load.assert_called_once()
                                    mock_import_downloader.assert_called_once_with(
                                        "nemo_curator.download.test_downloader"
                                    )
                                    mock_import_iterator.assert_called_once_with(
                                        "nemo_curator.download.test_iterator"
                                    )
                                    mock_import_extractor.assert_called_once_with(
                                        "nemo_curator.download.test_extractor"
                                    )
                                    mock_expand.assert_called_once_with(
                                        "/path/to/download"
                                    )
                                    mock_downloader_class.assert_called_once_with(
                                        download_dir="/path/to/download",
                                        param1="value1",
                                    )
                                    mock_iterator_class.assert_called_once_with(
                                        param2="value2"
                                    )
                                    mock_extractor_class.assert_called_once_with(
                                        param3="value3"
                                    )
                                    assert mock_locate_func.call_count == 2
                                    assert dataset_format == {
                                        "field1": str,
                                        "field2": int,
                                    }
                                    assert downloader == mock_downloader_instance
                                    assert iterator == mock_iterator_instance
                                    assert extractor == mock_extractor_instance

    def test_build_downloader_with_default_download_dir(self):
        """Test building a downloader with a default download directory."""
        # Create mock downloader config without download_dir
        downloader_params = {
            "download_module": "nemo_curator.download.test_downloader",
            "download_params": {"param1": "value1"},
            "iterator_module": "nemo_curator.download.test_iterator",
            "iterator_params": {"param2": "value2"},
            "extract_module": "nemo_curator.download.test_extractor",
            "extract_params": {"param3": "value3"},
            "format": {"field1": "str", "field2": "int"},
        }

        # Mock the open function and yaml.load
        with patch("builtins.open", mock_open(read_data="dummy")) as mock_file:
            with patch("yaml.load", return_value=downloader_params) as mock_load:
                # Mock the import functions and classes
                mock_downloader_class = MagicMock()
                mock_downloader_instance = MagicMock()
                mock_downloader_class.return_value = mock_downloader_instance

                mock_iterator_class = MagicMock()
                mock_iterator_instance = MagicMock()
                mock_iterator_class.return_value = mock_iterator_instance

                mock_extractor_class = MagicMock()
                mock_extractor_instance = MagicMock()
                mock_extractor_class.return_value = mock_extractor_instance

                # Mock locate for format types
                def mock_locate(type_name):
                    if type_name == "str":
                        return str
                    elif type_name == "int":
                        return int
                    return None

                with patch(
                    "nemo_curator.utils.config_utils.import_downloader",
                    return_value=mock_downloader_class,
                ) as mock_import_downloader:
                    with patch(
                        "nemo_curator.utils.config_utils.import_iterator",
                        return_value=mock_iterator_class,
                    ) as mock_import_iterator:
                        with patch(
                            "nemo_curator.utils.config_utils.import_extractor",
                            return_value=mock_extractor_class,
                        ) as mock_import_extractor:
                            with patch(
                                "nemo_curator.utils.config_utils.locate",
                                side_effect=mock_locate,
                            ) as mock_locate_func:
                                with patch(
                                    "nemo_curator.utils.config_utils.expand_outdir_and_mkdir"
                                ) as mock_expand:
                                    # Call the function with default download dir
                                    default_dir = "/default/download/dir"
                                    downloader, iterator, extractor, dataset_format = (
                                        build_downloader(
                                            "dummy_path.yaml",
                                            default_download_dir=default_dir,
                                        )
                                    )

                                    # Check assertions
                                    mock_expand.assert_called_once_with(default_dir)
                                    mock_downloader_class.assert_called_once_with(
                                        download_dir=default_dir, param1="value1"
                                    )

    def test_build_downloader_with_none_download_params(self):
        """Test building a downloader with None download_params."""
        # Create mock downloader config with None download_params
        downloader_params = {
            "download_module": "nemo_curator.download.test_downloader",
            "download_params": None,
            "iterator_module": "nemo_curator.download.test_iterator",
            "iterator_params": {"param2": "value2"},
            "extract_module": "nemo_curator.download.test_extractor",
            "extract_params": {"param3": "value3"},
            "format": {"field1": "str", "field2": "int"},
        }

        # Mock the open function and yaml.load
        with patch("builtins.open", mock_open(read_data="dummy")) as mock_file:
            with patch("yaml.load", return_value=downloader_params) as mock_load:
                # Mock the import functions and classes
                mock_downloader_class = MagicMock()
                mock_downloader_instance = MagicMock()
                mock_downloader_class.return_value = mock_downloader_instance

                mock_iterator_class = MagicMock()
                mock_iterator_instance = MagicMock()
                mock_iterator_class.return_value = mock_iterator_instance

                mock_extractor_class = MagicMock()
                mock_extractor_instance = MagicMock()
                mock_extractor_class.return_value = mock_extractor_instance

                # Mock locate for format types
                def mock_locate(type_name):
                    if type_name == "str":
                        return str
                    elif type_name == "int":
                        return int
                    return None

                with patch(
                    "nemo_curator.utils.config_utils.import_downloader",
                    return_value=mock_downloader_class,
                ) as mock_import_downloader:
                    with patch(
                        "nemo_curator.utils.config_utils.import_iterator",
                        return_value=mock_iterator_class,
                    ) as mock_import_iterator:
                        with patch(
                            "nemo_curator.utils.config_utils.import_extractor",
                            return_value=mock_extractor_class,
                        ) as mock_import_extractor:
                            with patch(
                                "nemo_curator.utils.config_utils.locate",
                                side_effect=mock_locate,
                            ) as mock_locate_func:
                                with patch(
                                    "nemo_curator.utils.config_utils.expand_outdir_and_mkdir"
                                ) as mock_expand:
                                    # Call the function with default download dir
                                    # This should throw a TypeError because download_params is None,
                                    # and we try to check if "download_dir" is in None
                                    with pytest.raises(TypeError):
                                        build_downloader(
                                            "dummy_path.yaml",
                                            default_download_dir="/default/download/dir",
                                        )
