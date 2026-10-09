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

"""Train the Omni-Fuse projection network and project raw embeddings."""

from __future__ import annotations

from utils import config_parser, load_tutorial_config, print_outputs, run_projection


def main() -> int:
    parser = config_parser(__doc__ or "")
    args = parser.parse_args()
    config = load_tutorial_config(args.config)
    task = run_projection(config)
    metadata = dict(getattr(task, "_metadata", {}) or {})
    print_outputs(
        {
            "run_dir": str(config.run_dir),
            "projection_model_path": metadata.get("projection_model_path"),
            "projection_metrics_path": metadata.get("projection_metrics_path"),
            "projected_embeddings_path": metadata.get("projected_embeddings_path"),
            "projection_records_path": metadata.get("projection_records_path"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
