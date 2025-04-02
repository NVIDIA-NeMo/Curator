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

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Union

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.modules.base import BaseDeduplicationModule
from nemo_curator.modules.config import FuzzyDuplicatesConfig
from nemo_curator.modules.fuzzy_dedup._mapbuckets import _MapBuckets
from nemo_curator.modules.fuzzy_dedup._shuffle import _Shuffle
from nemo_curator.modules.fuzzy_dedup.bucketstoedges import BucketsToEdges
from nemo_curator.modules.fuzzy_dedup.connectedcomponents import ConnectedComponents
from nemo_curator.modules.fuzzy_dedup.jaccardsimilarity import JaccardSimilarity
from nemo_curator.modules.fuzzy_dedup.lsh import LSH
from nemo_curator.modules.fuzzy_dedup.minhash import MinHash
from nemo_curator.modules.meta import Sequential
from nemo_curator.utils.distributed_utils import performance_report_if_with_ts_suffix
from nemo_curator.utils.duplicates_removal import remove_duplicates


class FuzzyDuplicates(BaseDeduplicationModule):
    def __init__(
        self,
        config: FuzzyDuplicatesConfig,
        logger: Union[logging.LoggerAdapter, str] = "./",
        perform_removal: bool = False,
    ):
        """
        Parameters
        ----------
        config: FuzzyDuplicatesConfig,
            Config options for finding FuzzyDuplicates
        logger: Existing logger to log to, or a path to a log directory.
        perform_removal: Whether to remove duplicates from the dataset.
            Default is False.
        Returns
        -------
        DocumentDataset containing IDs of all documents and the corresponding duplicate group
        they belong to. Documents in the same group are near duplicates.
        """
        super().__init__(
            id_field=config.id_field,
            text_field=config.text_field,
            input_backend="cudf",
            logger=logger,
            perform_removal=perform_removal,
            profile_dir=config.profile_dir,
            cache_dir=config.cache_dir,
        )

        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "FuzzyDuplicates.log"),
                name="FuzzyDuplicates",
            )
        else:
            self._logger = logger

        self.config = config
        self.minhash = MinHash(
            seed=self.config.seed,
            num_hashes=self.config.num_hashes,
            char_ngrams=self.config.char_ngrams,
            use_64bit_hash=self.config.use_64_bit_hash,
            logger=self._logger,
            id_field=self.config.id_field,
            text_field=self.config.text_field,
            profile_dir=self.config.profile_dir,
            cache_dir=self.config.cache_dir,
        )
        self.lsh = LSH(
            cache_dir=self.config.cache_dir,
            num_hashes=self.config.num_hashes,
            num_buckets=self.config.num_buckets,
            buckets_per_shuffle=self.config.buckets_per_shuffle,
            false_positive_check=self.config.false_positive_check,
            logger=self._logger,
            id_fields=[self.config.id_field],
            profile_dir=self.config.profile_dir,
        )

        if self.config.false_positive_check:
            self.map_buckets = _MapBuckets(
                id_fields=[self.config.id_field],
                text_field=self.config.text_field,
                logger=self._logger,
                num_anchors=self.config.num_anchors,
            )
            self.jaccard_shuffle = _Shuffle(
                id_fields=[self.config.id_field],
                text_field=self.config.text_field,
                logger=self._logger,
                profile_dir=self.config.profile_dir,
            )
            self.jaccard_compute = JaccardSimilarity(
                id_field=self.config.id_field,
                text_field=self.config.text_field,
                ngram_width=self.config.char_ngrams,
                anchor_id_fields=[
                    f"anchor_{i}_{self.config.id_field}"
                    for i in range(self.config.num_anchors)
                ],
            )
        else:
            self.buckets_to_edges = BucketsToEdges(
                cache_dir=self.config.cache_dir,
                id_fields=self.config.id_field,
                logger=self._logger,
                profile_dir=self.config.profile_dir,
            )

        jaccard_pairs_fname = (
            "jaccard_similarity_results.parquet"
            if self.config.false_positive_check
            else "_edges.parquet"
        )
        self.connected_components = ConnectedComponents(
            cache_dir=self.config.cache_dir,
            jaccard_pairs_path=os.path.join(self.config.cache_dir, jaccard_pairs_fname),
            id_column=self.config.id_field,
            jaccard_threshold=self.config.jaccard_threshold,
            logger=self._logger,
            profile_dir=self.config.profile_dir,
        )

    def identify_duplicates(
        self, dataset: DocumentDataset
    ) -> Optional[DocumentDataset]:
        """
        Parameters
        ----------
        dataset: DocumentDataset
            The input datset to compute FuzzyDuplicates. Must contain a text and unique id field.

        Returns
        -------
        DocumentDataset containing IDs of all documents and the corresponding duplicate group
        they belong to. Documents in the same group are near duplicates.
        """

        # Minhash + LSH
        stage_num = 1
        print(f"Stage {stage_num}: Starting Minhash + LSH computation")
        minhashLSH = Sequential([self.minhash, self.lsh])
        buckets_df = minhashLSH(dataset)
        print(f"Stage {stage_num}: Minhash + LSH complete!")
        if buckets_df is None:
            print(
                f"Stage {stage_num}: No potential duplicate documents found during LSH"
            )
            return None
        stage_num += 1

        if self.config.false_positive_check:
            # Map buckets to lower cardinality distribution
            print(f"Stage {stage_num} (False Positive Check): Starting Map_Buckets")
            t0 = time.time()
            mapped_buckets_w_anchors_path = os.path.join(
                self.config.cache_dir, "anchor_docs_with_bk.parquet"
            )
            with performance_report_if_with_ts_suffix(
                self.config.profile_dir,
                "map_buckets",
            ):
                ddf_mapped_buckets_w_anchors = (
                    self.map_buckets.map_buckets_with_anchors(
                        documents_df=dataset.df, buckets_df=buckets_df.df
                    )
                )
                ddf_mapped_buckets_w_anchors.to_parquet(
                    mapped_buckets_w_anchors_path, write_index=False, overwrite=True
                )
            self._logger.info(
                f"Time taken for Map_buckets : {time.time() - t0}s and output written at {mapped_buckets_w_anchors_path}"
            )

            print(f"Stage {stage_num} (False Postive Check): Map_Buckets Complete!")
            stage_num += 1

            # Shuffle documents based on mapped buckets
            print(f"Stage {stage_num} (False Postive Check): Shuffle docs")
            shuffled_docs_path = os.path.join(
                self.config.cache_dir, "shuffled_docs.parquet"
            )
            self.jaccard_shuffle.shuffle_docs_on_buckets(
                documents_df=dataset.df,
                bucket_w_anchors_path=mapped_buckets_w_anchors_path,
                output_shuffled_docs_path=shuffled_docs_path,
                bucket_mapping_df_blocksize=self.config.bucket_mapping_blocksize,
                parts_per_worker=self.config.parts_per_worker,
                bucket_parts_per_worker=self.config.bucket_parts_per_worker,
            )
            print(f"Stage {stage_num} (False Postive Check): Shuffle docs complete!")
            stage_num += 1

            # jaccard comparision within buckets
            print(
                f"Stage {stage_num} (False Postive Check): Jaccard Similarity in Buckets"
            )
            jaccard_pairs_path = os.path.join(
                self.config.cache_dir, "jaccard_similarity_results.parquet"
            )
            t0 = time.time()
            with performance_report_if_with_ts_suffix(
                self.config.profile_dir,
                "jaccard-similarity",
            ):
                jaccard_pairs_df = self.jaccard_compute.jaccard_compute(
                    shuffled_docs_path=shuffled_docs_path
                )
                jaccard_pairs_df.to_parquet(
                    jaccard_pairs_path,
                    write_index=False,
                    write_metadata_file=False,
                    overwrite=True,
                )
                self._logger.info(
                    f"Time taken for Jaccard Similarity = {time.time()-t0}s and output written at {jaccard_pairs_path}"
                )

            print(
                f"Stage {stage_num} (False Postive Check): Jaccard Similarity in Buckets Complete!"
            )
            stage_num += 1

        else:
            # Map buckets to lower cardinality distribution
            print(f"Stage {stage_num}: Starting LSH Buckets to Graph Edgelist")
            self.buckets_to_edges(buckets_df)
            print(
                f"Stage {stage_num}: Starting LSH Buckets to Graph Edgelist Complete!"
            )
            stage_num += 1

        # Connected components across buckets
        print(f"Stage {stage_num}: Connected Components across buckets")
        cc_path = os.path.join(self.config.cache_dir, "connected_components.parquet")
        self.connected_components.cc_workflow(cc_path)
        print(f"Stage {stage_num}: Connected Components across buckets complete!")
        stage_num += 1

        return DocumentDataset.read_parquet(
            cc_path,
            backend="cudf",
            # We read with files_per_partition=1 so that groups are read in whole (and do not exist across partitions)
            files_per_partition=1,
            blocksize=None,
        )

    def remove(
        self, dataset: DocumentDataset, duplicates_to_remove: Optional[DocumentDataset]
    ) -> DocumentDataset:
        """
        Remove fuzzy duplicates from a given DocumentDataset
        Parameters
        ----------
        dataset: DocumentDataset
          The input dataset from which to remove fuzzy duplicates
        duplicates_to_remove: DocumentDataset
          The dataset containing IDs of the fuzzy duplicates to remove
        Returns
        -------
        DocumentDataset containing only non-duplicate documents
        """

        if not duplicates_to_remove:
            print("No fuzzy duplicates to remove, returning original dataset")
            return dataset

        result = remove_duplicates(
            left=dataset.df,
            duplicates=duplicates_to_remove.df,
            id_field=self.config.id_field,
            group_field="group",
        )
        return DocumentDataset(result)
