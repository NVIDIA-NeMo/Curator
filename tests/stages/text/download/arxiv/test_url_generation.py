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

from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.stages.text.download.arxiv.url_generation import ArxivUrlGenerator


class TestArxivUrlGenerator:
    """Test suite for ArxivUrlGenerator."""

    @patch("subprocess.run")
    def test_successful_retrieval(self, mock_run: MagicMock) -> None:
        """Test successful retrieval of ArXiv URLs."""
        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        # The output of s5cmd has a specific format we need to match for indexing
        # This specifically needs items at position 3, 7, 11, etc. to be URLs
        mock_result.stdout = "DATE1 TIME1 SIZE1 s3://arxiv/src/arXiv_src_1804_001.tar DATE2 TIME2 SIZE2 s3://arxiv/src/arXiv_src_1805_001.tar DATE3 TIME3 SIZE3 s3://arxiv/src/arXiv_src_1806_001.tar"
        mock_run.return_value = mock_result

        # Call the function
        result = ArxivUrlGenerator().generate_urls()

        # Check subprocess call
        mock_run.assert_called_once()
        assert "s5cmd" in mock_run.call_args[0][0]
        assert "s3://arxiv/src/" in mock_run.call_args[0][0]

        # Check result URLs
        assert len(result) == 3
        assert result == [
            "s3://arxiv/src/arXiv_src_1804_001.tar",
            "s3://arxiv/src/arXiv_src_1805_001.tar",
            "s3://arxiv/src/arXiv_src_1806_001.tar",
        ]

    @patch("subprocess.run")
    def test_command_failure(self, mock_run: MagicMock) -> None:
        """Test error handling when s5cmd command fails."""
        # Mock subprocess result (failure)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Command failed: access denied"
        mock_run.return_value = mock_result

        # Call the function and check for error
        with pytest.raises(RuntimeError) as excinfo:
            ArxivUrlGenerator().generate_urls()

        # Check error message
        assert "Unable to get arxiv urls" in str(excinfo.value)
        assert "access denied" in str(excinfo.value)
