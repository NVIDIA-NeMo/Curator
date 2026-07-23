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

"""Build a sharded URL -> Lance row-id/row-address sidecar."""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path

from nemo_curator.stages.interleaved.lance import LanceTableConfig
from nemo_curator.stages.interleaved.lance.sidecar import build_sharded_sqlite_url_lance_sidecar


def aws_profile_storage_options(profile: str) -> dict[str, str]:
    cfg = configparser.ConfigParser()
    cfg.read(Path("~/.aws/config").expanduser())
    creds = configparser.ConfigParser()
    creds.read(Path("~/.aws/credentials").expanduser())

    out: dict[str, str] = {}
    if profile in creds:
        section = creds[profile]
        out["aws_access_key_id"] = section["aws_access_key_id"]
        out["aws_secret_access_key"] = section["aws_secret_access_key"]
        if "aws_session_token" in section:
            out["aws_session_token"] = section["aws_session_token"]

    config_section = None
    if f"profile {profile}" in cfg:
        config_section = cfg[f"profile {profile}"]
    elif profile in cfg:
        config_section = cfg[profile]
    if config_section is not None:
        if "endpoint_url" in config_section:
            out["aws_endpoint"] = config_section["endpoint_url"]
        if "region" in config_section:
            out["aws_region"] = config_section["region"]

    out.setdefault("aws_virtual_hosted_style_request", "false")
    out.setdefault("aws_region", "us-east-1")
    return out


def parse_storage_options(values: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            msg = f"storage option must be KEY=VALUE, got {value!r}"
            raise ValueError(msg)
        key, option_value = value.split("=", 1)
        if not key:
            msg = f"storage option key must be non-empty, got {value!r}"
            raise ValueError(msg)
        options[key] = option_value
    return options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image-uri", required=True, help="Lance image dataset URI or local path")
    parser.add_argument("--image-version", type=int, default=None, help="Pinned Lance version")
    parser.add_argument("--key-column", default="url", help="Image-table URL/key column")
    parser.add_argument("--output-dir", required=True, help="Sidecar output directory")
    parser.add_argument("--shard-count", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means scan all rows")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--insert-batch-rows", type=int, default=8192)
    parser.add_argument("--commit-every-rows", type=int, default=5_000_000)
    parser.add_argument("--progress-every-rows", type=int, default=1_000_000)
    parser.add_argument("--sample-url-count", type=int, default=32_768)
    parser.add_argument("--aws-profile", default="", help="Optional AWS profile to convert into Lance storage_options")
    parser.add_argument(
        "--storage-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Lance storage option. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    storage_options = parse_storage_options(args.storage_option)
    if args.aws_profile:
        storage_options = {**aws_profile_storage_options(args.aws_profile), **storage_options}

    report = build_sharded_sqlite_url_lance_sidecar(
        dataset=LanceTableConfig(
            uri=args.image_uri,
            version=args.image_version,
            storage_options=storage_options,
        ),
        output_dir=args.output_dir,
        key_column=args.key_column,
        shard_count=args.shard_count,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        insert_batch_rows=args.insert_batch_rows,
        commit_every_rows=args.commit_every_rows,
        progress_every_rows=args.progress_every_rows,
        sample_url_count=args.sample_url_count,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
