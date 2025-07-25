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
import argparse
import os

import psutil

from nemo_curator import __version__


class ArgumentHelper:
    """
    A helper class to add common arguments to an argparse.ArgumentParser instance.
    """

    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser

        version_string = f"NVIDIA NeMo Curator -- v{__version__}"
        self.attach_version_arg(version_string)
        parser_print_help = self.parser.print_help

        def print_help_with_version(*args, **kwargs) -> None:
            print(version_string)
            parser_print_help(*args, **kwargs)

        self.parser.print_help = print_help_with_version

    def attach_version_arg(self, version_string: str) -> None:
        self.parser.add_argument(
            "--version",
            "-v",
            action="version",
            version=version_string,
            help="Show the version and exit.",
        )

    @staticmethod
    def attach_bool_arg(
        parser: argparse.ArgumentParser,
        flag_name: str,
        default: bool = False,
        help: str | None = None,  # noqa: A002
    ) -> None:
        attr_name = flag_name.replace("-", "_")
        help = flag_name.replace("-", " ") if help is None else help  # noqa: A001
        parser.add_argument(
            f"--{flag_name}",
            dest=attr_name,
            action="store_true",
            help=help,
        )
        parser.add_argument(
            f"--no-{flag_name}",
            dest=attr_name,
            action="store_false",
            help=help,
        )

        parser.set_defaults(**{attr_name: default})

    def add_arg_batch_size(
        self,
        default: int = 64,
        help: str = "Number of files to read into memory at a time.",  # noqa: A002
    ) -> None:
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=default,
            help=help,
        )

    def add_arg_device(self) -> None:
        self.parser.add_argument(
            "--device",
            type=str,
            default="gpu",
            help='Device to run the script on. Either "cpu" or "gpu".',
        )

    def add_arg_enable_spilling(self) -> None:
        self.parser.add_argument("--enable-spilling", action="store_true")

    def add_arg_language(self, help: str) -> None:  # noqa: A002
        self.parser.add_argument(
            "--language",
            type=str,
            default="en",
            help=help,
        )

    def add_arg_log_dir(self, default: str) -> None:
        self.parser.add_argument(
            "--log-dir",
            type=str,
            default=default,
            help="The output log directory where node and local ranks will write their respective log files.",
        )

    def add_arg_input_data_dir(
        self,
        required: bool = False,
        help: str = "Input directory consisting of .jsonl files that are accessible "  # noqa: A002
        "to all nodes. Use this for a distributed file system.",
    ) -> None:
        self.parser.add_argument(
            "--input-data-dir",
            type=str,
            default=None,
            required=required,
            help=help,
        )

    def add_arg_input_file_type(
        self,
        choices: list | None = None,
        required: bool = False,
        help: str = "File type of the dataset to be read in. Supported file formats "  # noqa: A002
        'include "jsonl" (default), "pickle", or "parquet".',
    ) -> None:
        self.parser.add_argument(
            "--input-file-type",
            type=str,
            default="jsonl",
            required=required,
            choices=choices,
            help=help,
        )

    def add_arg_input_file_extension(
        self,
        help: str = "The file extension of the input files. If not provided, the input file type will be used.",  # noqa: A002
    ) -> None:
        self.parser.add_argument(
            "--input-file-extension",
            type=str,
            default=None,
            help=help,
        )

    def add_arg_input_local_data_dir(self) -> None:
        self.parser.add_argument(
            "--input-local-data-dir",
            type=str,
            default=None,
            help="Input directory consisting of dataset files. "
            "Use this argument when a distributed file system is not available.",
        )

    def add_arg_input_meta(self) -> None:
        self.parser.add_argument(
            "--input-meta",
            type=str,
            default=None,
            help="A string formatted as a dictionary, which outlines the field names and "
            "their respective data types within the JSONL input files.",
        )

    def add_arg_input_text_field(self) -> None:
        self.parser.add_argument(
            "--input-text-field",
            type=str,
            default="text",
            help="The name of the field within each datapoint object of the input file that contains the text.",
        )

    def add_arg_id_column(self) -> None:
        self.parser.add_argument(
            "--id-column",
            type=str,
            default="id",
            help="The name of the field within each datapoint object of the input file that contains the ID.",
        )

    def add_arg_id_column_type(self) -> None:
        self.parser.add_argument(
            "--id-column-type",
            type=str,
            default="int",
            help='The datatype of the ID field, either "int" or "str".',
        )

    def add_arg_minhash_length(self) -> None:
        self.parser.add_argument(
            "--minhash-length",
            type=int,
            default=260,
            help="The minhash signature length of each input document.",
        )

    def add_arg_nvlink_only(self) -> None:
        self.parser.add_argument(
            "--nvlink-only",
            action="store_true",
            help="Start a local cluster with only NVLink enabled."
            'Only applicable when protocol="ucx" and no scheduler file or address is specified.',
        )

    def add_arg_output_data_dir(self, help: str) -> None:  # noqa: A002
        self.parser.add_argument(
            "--output-data-dir",
            type=str,
            required=True,
            help=help,
        )

    def add_arg_output_dir(
        self,
        required: bool = False,
        help: str = "The output directory to write results.",  # noqa: A002
    ) -> None:
        self.parser.add_argument(
            "--output-dir",
            type=str,
            required=required,
            help=help,
        )

    def add_arg_output_file_type(
        self,
        choices: list | None = None,
        help: str = "File type the dataset will be written to. Supported file formats "  # noqa: A002
        'include "jsonl" (default), "pickle", or "parquet".',
    ) -> None:
        self.parser.add_argument(
            "--output-file-type",
            type=str,
            default="jsonl",
            choices=choices,
            help=help,
        )

    def add_arg_output_train_file(self, help: str, default: str | None = None) -> None:  # noqa: A002
        self.parser.add_argument(
            "--output-train-file",
            type=str,
            default=default,
            help=help,
        )

    def add_arg_protocol(self) -> None:
        self.parser.add_argument(
            "--protocol",
            type=str,
            default="ucx",
            help="Protocol to use for Dask cluster. "
            "Note: This only applies to the LocalCUDACluster. If providing a user-created "
            "cluster, please refer to "
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-protocol.",
        )

    def add_arg_rmm_pool_size(self) -> None:
        self.parser.add_argument(
            "--rmm-pool-size",
            type=str,
            default="14GB",
            help="Initial pool size to use for the RMM Pool Memory allocator. "
            "Note: This only applies to the LocalCUDACluster. If providing a user-created "
            "cluster, please refer to "
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-rmm-pool-size.",
        )

    def add_arg_scheduler_address(self) -> None:
        self.parser.add_argument(
            "--scheduler-address",
            type=str,
            default=None,
            help="Address to the scheduler of a created Dask cluster. If not provided, "
            "a single-node LocalCUDACluster will be started.",
        )

    def add_arg_scheduler_file(self) -> None:
        self.parser.add_argument(
            "--scheduler-file",
            type=str,
            default=None,
            help="Path to the scheduler file of a created Dask cluster. If not provided, "
            " a single-node LocalCUDACluster will be started.",
        )

    def add_arg_seed(self, default: int = 42, help: str = "If specified, the random seed used for shuffling.") -> None:  # noqa: A002
        self.parser.add_argument(
            "--seed",
            type=int,
            default=default,
            help=help,
        )

    def add_arg_set_torch_to_use_rmm(self) -> None:
        self.parser.add_argument("--set-torch-to-use-rmm", action="store_true")

    def add_arg_shuffle(self, help: str) -> None:  # noqa: A002
        ArgumentHelper.attach_bool_arg(
            self.parser,
            "shuffle",
            help=help,
        )

    def add_arg_text_ddf_blocksize(self) -> None:
        self.parser.add_argument(
            "--text-ddf-blocksize",
            type=int,
            default=256,
            help="The block size for chunking JSONL files for text DataFrames in MB.",
        )

    def add_arg_model_path(self, help: str = "The path to the model file") -> None:  # noqa: A002
        self.parser.add_argument(
            "--pretrained-model-name-or-path",
            type=str,
            help=help,
            required=False,
        )

    def add_arg_max_mem_gb_classifier(self) -> None:
        self.parser.add_argument(
            "--max-mem-gb-classifier",
            default=None,
            type=int,
            help="Specify the maximum GPU memory (in GB) for the classifier. "
            "Defaults to using the total GPU memory minus 4 GB if not specified.",
        )

    def add_arg_autocast(self, help: str = "Whether to use autocast or not") -> None:  # noqa: A002
        ArgumentHelper.attach_bool_arg(
            parser=self.parser,
            flag_name="autocast",
            default=True,
            help=help,
        )

    def add_arg_max_chars(self, default: int = 2000) -> None:
        self.parser.add_argument(
            "--max-chars",
            type=int,
            default=default,
            help="Truncates all documents in the dataset to this number of characters before running model inference on them.",
        )

    def add_distributed_args(self) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are needed for Dask cluster setup.

        """
        self.parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help='Device to run the script on. Either "cpu" or "gpu".',
        )
        self.parser.add_argument(
            "--files-per-partition",
            type=int,
            default=2,
            help="Number of JSONL files to combine into a single partition.",
        )
        self.parser.add_argument(
            "--n-workers",
            type=int,
            default=os.cpu_count(),
            help="The number of workers to run in total on the Dask CPU cluster.",
        )
        self.parser.add_argument(
            "--num-files",
            type=int,
            default=None,
            help="Upper limit on the number of JSON files to process.",
        )
        self.parser.add_argument(
            "--nvlink-only",
            action="store_true",
            help="Start a local cluster with only NVLink enabled. "
            'Only applicable when protocol="ucx" and no scheduler file or address is specified.',
        )
        self.parser.add_argument(
            "--protocol",
            type=str,
            default="tcp",
            help="Protcol to use for Dask cluster"
            "Note: This only applies to the LocalCUDACluster. If providing a user-created "
            "cluster, please refer to"
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-protocol.",
        )
        self.parser.add_argument(
            "--rmm-pool-size",
            type=str,
            default=None,
            help="Initial pool size to use for the RMM Pool Memory allocator. "
            "Note: This only applies to the LocalCUDACluster. If providing a user-created "
            "cluster, please refer to"
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-rmm-pool-size.",
        )
        self.parser.add_argument(
            "--scheduler-address",
            type=str,
            default=None,
            help="Address to the scheduler of a created Dask cluster. If not provided, "
            "a single-node cluster will be started.",
        )
        self.parser.add_argument(
            "--scheduler-file",
            type=str,
            default=None,
            help="Path to the scheduler file of a created Dask cluster. If not provided, "
            " a single-node cluster will be started.",
        )
        self.parser.add_argument(
            "--threads-per-worker",
            type=int,
            default=1,
            help="The number of threads to launch per worker on the Dask CPU cluster. Usually best to set at 1 due to the GIL.",
        )

        return self.parser

    def set_default_n_workers(self, max_mem_gb_per_worker: float) -> None:
        """
        Sets the default --n-workers for a script to maximize parallelization while
        ensuring we don't trigger an out of memory error. Like --n-workers, this
        only applies when running the script locally.

        Args:
            max_mem_per_worker (float): The maximum memory that each worker usually achieves for a script
                in units of gigabytes. It can be determined by watching the Dask dashboard. This value may
                change based on the size of each shard, so use a JSONL shard size of about 100 MB.
        """
        cpu_worker_limit = os.cpu_count()

        memory_gb = psutil.virtual_memory().total / (1024**3)
        mem_worker_limit = memory_gb // max_mem_gb_per_worker

        n_workers = min(cpu_worker_limit, mem_worker_limit)
        self.parser.set_defaults(n_workers=n_workers)

    @staticmethod
    def parse_client_args(args: argparse.Namespace) -> dict:
        """
        Extracts relevant arguments from an argparse namespace to pass to get_client.

        """
        relevant_args = [
            "scheduler_address",
            "scheduler_file",
            "n_workers",
            "threads_per_worker",
            "nvlink_only",
            "protocol",
            "rmm_pool_size",
            "enable_spilling",
            "set_torch_to_use_rmm",
        ]
        dict_args = vars(args)

        parsed_args = {arg: dict_args[arg] for arg in relevant_args if arg in dict_args}
        if "device" in dict_args:
            parsed_args["cluster_type"] = dict_args["device"]

        return parsed_args

    @staticmethod
    def parse_distributed_classifier_args(
        description: str = "Default distributed classifier argument parser.",
        max_chars_default: int = 2000,
    ) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are common to multiple stages
        of the pipeline.

        """
        parser = argparse.ArgumentParser(
            description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        arg_helper = ArgumentHelper(parser)
        arg_helper.add_distributed_classifier_cluster_args()
        arg_helper.add_arg_input_data_dir(required=True)
        arg_helper.add_arg_output_data_dir(help="The path of the output files.")
        arg_helper.add_arg_input_file_type()
        arg_helper.add_arg_input_file_extension()
        arg_helper.add_arg_output_file_type()
        arg_helper.add_arg_input_text_field()
        arg_helper.add_arg_batch_size(help="The batch size to be used for inference.")
        arg_helper.add_arg_model_path()
        arg_helper.add_arg_autocast()
        arg_helper.add_arg_max_chars(default=max_chars_default)

        return arg_helper.parser

    def add_distributed_classifier_cluster_args(self) -> None:
        """
        Adds Dask cluster arguments needed for the distributed data classifiers.
        """
        self.add_distributed_args()
        self.add_arg_enable_spilling()
        self.add_arg_set_torch_to_use_rmm()
        self.add_arg_max_mem_gb_classifier()

        # Set low default RMM pool size for classifier
        # to allow pytorch to grow its memory usage
        # by default
        self.parser.set_defaults(rmm_pool_size="512MB")
        # Setting to False makes it more stable for long running jobs
        # possibly because of memory fragmentation
        self.parser.set_defaults(set_torch_to_use_rmm=False)

    def parse_gpu_dedup_args(self) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are common to multiple stages
        of the fuzzy deduplication pipeline.

        """

        self.add_distributed_args()

        # Set default device to GPU for dedup
        self.parser.set_defaults(device="gpu")
        self.parser.set_defaults(set_torch_to_use_rmm=False)
        self.parser.add_argument(
            "--input-data-dirs",
            type=str,
            nargs="+",
            default=None,
            help="Input directories consisting of .jsonl files that are accessible "
            "to all nodes. This path must be accessible by all machines in the cluster.",
        )
        self.parser.add_argument(
            "--input-json-text-field",
            type=str,
            default="text",
            help="The name of the field within each JSON object of the JSONL "
            "file that contains the text from which minhashes will be computed.",
        )
        self.parser.add_argument(
            "--input-json-id-field",
            type=str,
            required=True,
            help="The name of the field within each JSON object of the JSONL "
            "file that assigns a unqiue ID to each document. "
            "Can be created by running the script "
            "../scripts/add_id.py, which adds the field "
            "to the documents in a distributed fashion.",
        )
        self.parser.add_argument(
            "--log-dir",
            type=str,
            default="./logs/",
            help="The output log directory.",
        )
        self.parser.add_argument(
            "--profile-path",
            type=str,
            default=None,
            help="Path to save Dask profile.",
        )

    @staticmethod
    def parse_semdedup_args(
        description: str = "Default argument parser for semantic deduplication.",
    ) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are common to multiple stages of the semantic deduplication pipeline.

        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=description,
        )
        arg_helper = ArgumentHelper(parser)
        arg_helper.add_distributed_args()

        arg_helper.add_arg_input_data_dir()
        arg_helper.add_arg_input_file_extension()
        arg_helper.add_arg_input_file_type()
        arg_helper.add_arg_input_text_field()
        arg_helper.add_arg_id_column()

        arg_helper.parser.add_argument(
            "--config-file",
            type=str,
            help="Path to the semantic deduplication configuration file.",
            required=True,
        )
        # Set low default RMM pool size for classifier
        # to allow pytorch to grow its memory usage
        # by default
        parser.set_defaults(rmm_pool_size="512MB")
        parser.set_defaults(device="gpu")
        parser.set_defaults(set_torch_to_use_rmm=False)

        return parser
