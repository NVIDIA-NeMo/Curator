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

import argparse

from nemo_curator.utils.script_utils import ArgumentHelper


class TestArgumentHelper:
    def test_argument_helper(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        assert "--version" in argument_helper.parser.format_help()

    def test_attach_bool_arg(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.attach_bool_arg(
            argument_helper.parser, "test", default=False, help="test help"
        )
        assert "--test" in parser.format_help()
        assert "test help" in parser.format_help()
        assert "--no-test" in parser.format_help()

    def test_add_arg_batch_size(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_batch_size()
        assert "--batch-size" in argument_helper.parser.format_help()

    def test_add_arg_device(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_device()
        assert "--device" in argument_helper.parser.format_help()

    def test_add_arg_enable_spilling(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_enable_spilling()
        assert "--enable-spilling" in argument_helper.parser.format_help()

    def test_add_arg_language(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        help = "The language of the dataset to be read in."
        argument_helper.add_arg_language(help=help)
        assert "--language" in argument_helper.parser.format_help()
        assert help in argument_helper.parser.format_help()

    def test_add_arg_log_dir(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        default = "./log"
        argument_helper.add_arg_log_dir(default=default)
        assert "--log-dir" in argument_helper.parser.format_help()

    def test_add_arg_input_data_dir(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_input_data_dir()
        assert "--input-data-dir" in argument_helper.parser.format_help()

    def test_add_arg_input_file_type(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        choices = ["jsonl", "pickle", "parquet"]
        argument_helper.add_arg_input_file_type(choices=choices)
        assert "--input-file-type" in argument_helper.parser.format_help()
        assert "jsonl" in argument_helper.parser.format_help()
        assert "pickle" in argument_helper.parser.format_help()
        assert "parquet" in argument_helper.parser.format_help()

    def test_add_arg_input_file_extension(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_input_file_extension()
        assert "--input-file-extension" in argument_helper.parser.format_help()

    def test_add_arg_input_local_data_dir(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_input_local_data_dir()
        assert "--input-local-data-dir" in argument_helper.parser.format_help()

    def test_add_arg_input_meta(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_input_meta()
        assert "--input-meta" in argument_helper.parser.format_help()

    def test_add_arg_input_text_field(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_input_text_field()
        assert "--input-text-field" in argument_helper.parser.format_help()

    def test_add_arg_id_column(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_id_column()
        assert "--id-column" in argument_helper.parser.format_help()

    def test_add_arg_id_column_type(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_id_column_type()
        assert "--id-column-type" in argument_helper.parser.format_help()

    def test_add_arg_minhash_length(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_minhash_length()
        assert "--minhash-length" in argument_helper.parser.format_help()

    def test_add_arg_nvlink_only(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_nvlink_only()
        assert "--nvlink-only" in argument_helper.parser.format_help()

    def test_add_arg_output_data_dir(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        help = "Output data directory."
        argument_helper.add_arg_output_data_dir(help=help)
        assert "--output-data-dir" in argument_helper.parser.format_help()
        assert help in argument_helper.parser.format_help()

    def test_add_arg_output_dir(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_output_dir()
        assert "--output-dir" in argument_helper.parser.format_help()

    def test_add_arg_output_file_type(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        choices = ["jsonl", "pickle", "parquet"]
        argument_helper.add_arg_output_file_type(choices=choices)
        assert "--output-file-type" in argument_helper.parser.format_help()
        assert "jsonl" in argument_helper.parser.format_help()
        assert "pickle" in argument_helper.parser.format_help()
        assert "parquet" in argument_helper.parser.format_help()

    def test_add_arg_output_train_file(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        help = "The output train file."
        argument_helper.add_arg_output_train_file(help=help)
        assert "--output-train-file" in argument_helper.parser.format_help()
        assert help in argument_helper.parser.format_help()

    def test_add_arg_protocol(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_protocol()
        assert "--protocol" in argument_helper.parser.format_help()

    def test_add_arg_rmm_pool_size(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_rmm_pool_size()
        assert "--rmm-pool-size" in argument_helper.parser.format_help()

    def test_add_arg_scheduler_address(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_scheduler_address()
        assert "--scheduler-address" in argument_helper.parser.format_help()

    def test_add_arg_scheduler_file(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_scheduler_file()
        assert "--scheduler-file" in argument_helper.parser.format_help()

    def test_add_arg_seed(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_seed()
        assert "--seed" in argument_helper.parser.format_help()

    def test_add_arg_set_torch_to_use_rmm(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_set_torch_to_use_rmm()
        assert "--set-torch-to-use-rmm" in argument_helper.parser.format_help()

    def test_add_arg_shuffle(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        help = "Shuffle argument help"
        argument_helper.add_arg_shuffle(help=help)
        assert "--shuffle" in argument_helper.parser.format_help()
        assert help in argument_helper.parser.format_help()

    def test_add_arg_text_ddf_blocksize(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_text_ddf_blocksize()
        assert "--text-ddf-blocksize" in argument_helper.parser.format_help()

    def test_add_arg_model_path(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_model_path()
        assert "--pretrained-model-name-or-path" in argument_helper.parser.format_help()

    def test_add_arg_max_mem_gb_classifier(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_max_mem_gb_classifier()
        assert "--max-mem-gb-classifier" in argument_helper.parser.format_help()

    def test_add_arg_autocast(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_autocast()
        assert "--autocast" in argument_helper.parser.format_help()

    def test_add_arg_max_chars(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_arg_max_chars()
        assert "--max-chars" in argument_helper.parser.format_help()

    def test_distributed_args(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_distributed_args()

        assert "--device" in argument_helper.parser.format_help()
        assert "--files-per-partition" in argument_helper.parser.format_help()
        assert "--n-workers" in argument_helper.parser.format_help()
        assert "--num-files" in argument_helper.parser.format_help()
        assert "--nvlink-only" in argument_helper.parser.format_help()
        assert "--protocol" in argument_helper.parser.format_help()
        assert "--rmm-pool-size" in argument_helper.parser.format_help()
        assert "--scheduler-address" in argument_helper.parser.format_help()
        assert "--scheduler-file" in argument_helper.parser.format_help()
        assert "--threads-per-worker" in argument_helper.parser.format_help()

    def test_set_default_n_workers(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        max_mem_gb_per_worker = 10.0
        argument_helper.set_default_n_workers(max_mem_gb_per_worker)
        assert "n_workers" in argument_helper.parser._defaults

    def test_parse_client_args(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        args = argparse.Namespace(device="gpu", n_workers=10, random_arg="abc")
        parsed_args = argument_helper.parse_client_args(args)
        assert parsed_args["cluster_type"] == "gpu"
        assert parsed_args["n_workers"] == 10
        assert "random_arg" not in parsed_args

    def test_parse_distributed_classifier_args(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        parser = argument_helper.parse_distributed_classifier_args()

        assert "--device" in parser.format_help()
        assert "--files-per-partition" in parser.format_help()
        assert "--n-workers" in parser.format_help()
        assert "--num-files" in parser.format_help()
        assert "--nvlink-only" in parser.format_help()
        assert "--protocol" in parser.format_help()
        assert "--rmm-pool-size" in parser.format_help()
        assert "--scheduler-address" in parser.format_help()
        assert "--scheduler-file" in parser.format_help()
        assert "--threads-per-worker" in parser.format_help()
        assert "--enable-spilling" in parser.format_help()
        assert "--set-torch-to-use-rmm" in parser.format_help()
        assert "--max-mem-gb-classifier" in parser.format_help()
        assert "rmm_pool_size" in parser._defaults
        assert "set_torch_to_use_rmm" in parser._defaults

        assert "--input-data-dir" in parser.format_help()
        assert "--output-data-dir" in parser.format_help()
        assert "--input-file-type" in parser.format_help()
        assert "--input-file-extension" in parser.format_help()
        assert "--output-file-type" in parser.format_help()
        assert "--input-text-field" in parser.format_help()
        assert "--batch-size" in parser.format_help()
        assert "--pretrained-model-name-or-path" in parser.format_help()
        assert "--autocast" in parser.format_help()
        assert "--max-chars" in parser.format_help()

    def test_add_distributed_classifier_cluster_args(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.add_distributed_classifier_cluster_args()

        assert "--device" in argument_helper.parser.format_help()
        assert "--files-per-partition" in argument_helper.parser.format_help()
        assert "--n-workers" in argument_helper.parser.format_help()
        assert "--num-files" in argument_helper.parser.format_help()
        assert "--nvlink-only" in argument_helper.parser.format_help()
        assert "--protocol" in argument_helper.parser.format_help()
        assert "--rmm-pool-size" in argument_helper.parser.format_help()
        assert "--scheduler-address" in argument_helper.parser.format_help()
        assert "--scheduler-file" in argument_helper.parser.format_help()
        assert "--threads-per-worker" in argument_helper.parser.format_help()
        assert "--enable-spilling" in argument_helper.parser.format_help()
        assert "--set-torch-to-use-rmm" in argument_helper.parser.format_help()
        assert "--max-mem-gb-classifier" in argument_helper.parser.format_help()
        assert "rmm_pool_size" in argument_helper.parser._defaults
        assert "set_torch_to_use_rmm" in argument_helper.parser._defaults

    def test_parse_gpu_dedup_args(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        argument_helper.parse_gpu_dedup_args()

        assert "--device" in argument_helper.parser.format_help()
        assert "--files-per-partition" in argument_helper.parser.format_help()
        assert "--n-workers" in argument_helper.parser.format_help()
        assert "--num-files" in argument_helper.parser.format_help()
        assert "--nvlink-only" in argument_helper.parser.format_help()
        assert "--protocol" in argument_helper.parser.format_help()
        assert "--rmm-pool-size" in argument_helper.parser.format_help()
        assert "--scheduler-address" in argument_helper.parser.format_help()
        assert "--scheduler-file" in argument_helper.parser.format_help()
        assert "--threads-per-worker" in argument_helper.parser.format_help()

        assert "device" in argument_helper.parser._defaults
        assert "set_torch_to_use_rmm" in argument_helper.parser._defaults

        assert "--input-data-dirs" in argument_helper.parser.format_help()
        assert "--input-json-text-field" in argument_helper.parser.format_help()
        assert "--input-json-id-field" in argument_helper.parser.format_help()
        assert "--log-dir" in argument_helper.parser.format_help()
        assert "--profile-path" in argument_helper.parser.format_help()

    def test_parse_semdedup_args(self):
        parser = argparse.ArgumentParser()
        argument_helper = ArgumentHelper(parser)
        parser = argument_helper.parse_semdedup_args()

        assert "--device" in parser.format_help()
        assert "--files-per-partition" in parser.format_help()
        assert "--n-workers" in parser.format_help()
        assert "--num-files" in parser.format_help()
        assert "--nvlink-only" in parser.format_help()
        assert "--protocol" in parser.format_help()
        assert "--rmm-pool-size" in parser.format_help()
        assert "--scheduler-address" in parser.format_help()
        assert "--scheduler-file" in parser.format_help()
        assert "--threads-per-worker" in parser.format_help()

        assert "--input-data-dir" in parser.format_help()
        assert "--input-file-extension" in parser.format_help()
        assert "--input-file-type" in parser.format_help()
        assert "--input-text-field" in parser.format_help()
        assert "--id-column" in parser.format_help()

        assert "--config-file" in parser.format_help()

        assert "rmm_pool_size" in parser._defaults
        assert "device" in parser._defaults
        assert "set_torch_to_use_rmm" in parser._defaults
