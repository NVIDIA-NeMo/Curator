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

from ray_curator.utils.file_utils import infer_dataset_name_from_path


class TestInferDatasetNameFromPath:
    """Test cases for infer_dataset_name_from_path function."""

    def test_local_paths(self):
        """Test local path dataset name inference."""
        assert infer_dataset_name_from_path("/home/user/my_dataset/file.txt") == "my_dataset"
        assert infer_dataset_name_from_path("./file.txt") == "file"
        assert infer_dataset_name_from_path("file.txt") == "file"

    def test_cloud_paths(self):
        """Test cloud storage path dataset name inference."""
        assert infer_dataset_name_from_path("s3://bucket/datasets/my_dataset/") == "my_dataset"
        assert infer_dataset_name_from_path("s3://bucket/datasets/my_dataset/data.parquet") == "data.parquet"
        assert infer_dataset_name_from_path("s3://my-bucket") == "my-bucket"
        assert infer_dataset_name_from_path("abfs://container@account.dfs.core.windows.net/dataset") == "dataset"

    def test_case_conversion(self):
        """Test that results are converted to lowercase."""
        assert infer_dataset_name_from_path("s3://bucket/MyDataSet") == "mydataset"
        assert infer_dataset_name_from_path("/home/user/MyDataSet/file.txt") == "mydataset"
