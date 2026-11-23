# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""List the web images that interleaved pages link but do not hold, so they can be fetched."""

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, InterleavedBatch


@dataclass
class InterleavedRemoteImageUrlStage(ProcessingStage[InterleavedBatch, DocumentBatch]):
    """Interleaved rows -> one row per (sample, remote image URL).

    An image row is remote when nothing has stored its bytes: its ``source_ref`` path is still an http(s) URL,
    or it has no ``source_ref`` and ``url_field`` (a user column) holds the URL it was linked by.
    Output columns are ``sample_id``, ``image_url`` and each of ``sample_fields`` taken from the sample's
    metadata row (for example the page URL or a page score). An image repeated within a sample is listed once.
    """

    sample_fields: tuple[str, ...] = ()
    url_field: str | None = None
    name: str = "interleaved_remote_image_urls"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sample_id", "image_url", *self.sample_fields]

    def process(self, task: InterleavedBatch) -> DocumentBatch:
        t = task.to_pyarrow()
        images = t.filter(pc.equal(t["modality"], "image"))
        refs = images["source_ref"].to_pylist()
        linked = images[self.url_field].to_pylist() if self.url_field else [None] * len(refs)
        pairs = zip(refs, linked, strict=True)
        paths = [InterleavedBatch.parse_source_ref(ref)["path"] if ref else url for ref, url in pairs]
        remote = [p if isinstance(p, str) and p.startswith(("http://", "https://")) else None for p in paths]
        urls = pa.array(remote, pa.large_string())
        links = pa.table({"sample_id": images["sample_id"], "image_url": urls}).filter(pc.is_valid(urls))
        links = links.group_by(["sample_id", "image_url"]).aggregate([])
        samples = t.filter(pc.equal(t["modality"], "metadata")).select(["sample_id", *self.sample_fields])
        out = links.join(samples, keys="sample_id", join_type="left outer") if self.sample_fields else links
        return DocumentBatch(
            dataset_name=task.dataset_name, data=out, _metadata=task._metadata, _stage_perf=task._stage_perf
        )
