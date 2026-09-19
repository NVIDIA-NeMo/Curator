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

"""Run sharded SNS and EEE concurrently across multiple GPUs."""

from __future__ import annotations

from omnifuse_tutorial.parallel import run_parallel_sns_eee
from utils import config_parser, load_tutorial_config, print_outputs


def main() -> int:
    parser = config_parser(__doc__ or "")
    args = parser.parse_args()
    config = load_tutorial_config(args.config)
    print_outputs(run_parallel_sns_eee(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
