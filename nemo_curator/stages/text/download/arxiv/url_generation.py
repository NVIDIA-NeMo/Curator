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

import subprocess
from dataclasses import dataclass

from nemo_curator.stages.text.download import URLGenerator


@dataclass
class ArxivUrlGenerator(URLGenerator):
    """Generates URLs for Arxiv data."""

    def generate_urls(self) -> list[str]:
        return self._get_arxiv_urls()

    def _get_arxiv_urls(self) -> list[str]:
        command = "s5cmd --request-payer=requester ls s3://arxiv/src/ | grep '.tar'"
        result = subprocess.run(command, capture_output=True, text=True, shell=True, check=False)  # noqa: S602

        if result.returncode != 0:
            msg = f"Unable to get arxiv urls: {result.stderr}"
            raise RuntimeError(msg)

        urls = result.stdout.split()[3::4]
        urls.sort()

        return urls
