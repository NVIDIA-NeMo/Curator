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

import csv
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import dask_cudf

import dask.dataframe as dd
import pandas as pd

from nemo_curator.datasets.doc_dataset import DocumentDataset
from nemo_curator.utils.distributed_utils import _resolve_filename_col, write_to_disk
from nemo_curator.utils.file_utils import remove_path_extension
from nemo_curator.utils.import_utils import gpu_only_import

cudf = gpu_only_import("cudf")


class ParallelDataset(DocumentDataset):
    """
    An extension of the standard `DocumentDataset` with a special method that loads simple bitext.

    For data with more complicated metadata, please convert your data into jsonl/parquet/pickle format
    and use interfaces defined in `DocumentDataset`.
    """

    def persist(self) -> "ParallelDataset":
        return ParallelDataset(self.df.persist())

    @classmethod
    def read_simple_bitext(  # noqa: PLR0913
        cls,
        src_input_files: str | list[str],
        tgt_input_files: str | list[str],
        src_lang: str,
        tgt_lang: str,
        backend: str = "pandas",
        add_filename: bool | str = False,
        npartitions: int = 16,
    ) -> "ParallelDataset":
        """See `read_single_simple_bitext_file_pair` docstring for what "simple_bitext" means and usage of other parameters.

        Args:
            src_input_files (Union[str, List[str]]): one or several input files, in source language
            tgt_input_files (Union[str, List[str]]): one or several input files, in target language

        Raises:
            TypeError: If types of `src_input_files` and `tgt_input_files` doesn't agree.

        Returns:
            ParallelDataset: A `ParallelDataset` object with `self.df` holding the ingested simple bitext.
        """

        if isinstance(src_input_files, str) and isinstance(tgt_input_files, str):
            src_input_files = [src_input_files]
            tgt_input_files = [tgt_input_files]
        elif not isinstance(src_input_files, list) or not isinstance(tgt_input_files, list):
            msg = "Both file inputs must be strings or lists."
            raise TypeError(msg)

        # use default doc id for now
        # but in the future it might be useful to allow customizing doc id by passing a prefix
        df_files = []
        # We do not use `dd.from_map` because an individual file could be pretty large,
        # hence, it's not appropriate to partition based on individual files.
        # What we do is that we concatenate all the individual files and perform repartition.
        for src_input_file, tgt_input_file in zip(src_input_files, tgt_input_files, strict=False):
            df_file = ParallelDataset.read_single_simple_bitext_file_pair(
                (src_input_file, tgt_input_file),
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                backend=backend,
                add_filename=add_filename,
            )
            df_files.append(df_file)

        df = cudf if backend == "cudf" else pd

        data = dd.from_pandas(df.concat(df_files), npartitions=npartitions)
        return cls(data)

    def to_bitext(
        self,
        output_file_dir: str,
        write_to_filename: bool | str = False,
    ) -> None:
        """See `nemo_curator.utils.distributed_utils.write_to_disk` docstring for parameter usage."""
        write_to_disk(
            df=self.df,
            output_path=output_file_dir,
            write_to_filename=write_to_filename,
            output_type="bitext",
        )

    @staticmethod
    def read_single_simple_bitext_file_pair(  # noqa: PLR0913
        input_file_pair: tuple[str],
        src_lang: str,
        tgt_lang: str,
        doc_id: str | None = None,
        backend: str = "cudf",
        add_filename: bool | str = False,
    ) -> Union[dd.DataFrame, "dask_cudf.DataFrame"]:
        """This function reads a pair of "simple bitext" files into a pandas DataFrame.
        A simple bitext is a commonly data format in machine translation.
        It consists of two plain text files with the same number of lines, each line pair being translations of each other. For example:

        data.de:

        ```
        Wir besitzen keine Reisetaschen aus Leder.
        Die Firma produziert Computer für den deutschen Markt.
        ...
        ```

        data.en:

        ```
        We don't own duffel bags made of leather.
        The company produces computers for the German market.
        ...
        ```

        For simplicity, we also assume that the names of the two text files have the same prefix, except for different language code at the end as file extensions.

        Args:
            input_file_pair (Tuple[str]): A pair of file paths pointing to the input files
            src_lang (str): Source language, in ISO-639-1 (two character) format (e.g. 'en')
            tgt_lang (str): Target language, in ISO-639-1 (two character) format (e.g. 'en')
            doc_id (str, optional): A string document id to assign to every segment in the file. Defaults to None.
            backend (str, optional): Backend of the data frame. Defaults to "cudf".
            add_filename (Union[bool, str]): Whether to add a filename column to the DataFrame.
                If True, a new column is added to the DataFrame called `file_name`.
                If str, sets new column name. Default is False.


        Returns:
            Union[dd.DataFrame, dask_cudf.DataFrame]
        """
        src_input_file, tgt_input_file = input_file_pair
        if remove_path_extension(src_input_file) != remove_path_extension(tgt_input_file):
            msg = f"Assuming source and target filenames would have common prefix before language code, but got {src_input_file} and {tgt_input_file}."
            raise ValueError(msg)

        if not doc_id:
            doc_id = f"{src_input_file}▁{tgt_input_file}"

        df = cudf if backend == "cudf" else pd

        df_src = df.read_csv(src_input_file, names=["src"], sep="\t", quoting=csv.QUOTE_NONE)
        df_tgt = df.read_csv(tgt_input_file, names=["tgt"], sep="\t", quoting=csv.QUOTE_NONE)
        if len(df_src) != len(df_tgt):
            msg = f"We assume the source and target file would have the same number of lines, but got {len(df_src)} and {len(df_tgt)}."
            raise ValueError(msg)
        df_combined = df.concat([df_src, df_tgt], axis=1)
        df_combined["doc_id"] = doc_id
        df_combined["src_lang"] = src_lang
        df_combined["tgt_lang"] = tgt_lang

        if add_filename:
            df_combined[_resolve_filename_col(add_filename)] = remove_path_extension(src_input_file)

        return df_combined
