#!/usr/bin/env python3
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
"""Return success when a GitHub files JSON array contains a matching path."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def matches_file_payload(payload: object, pattern: str, path_key: str = "filename") -> bool:
    if not isinstance(payload, list):
        msg = "files payload must be a JSON array"
        raise TypeError(msg)
    path_regex = re.compile(pattern)
    return any(path_regex.search(str(item.get(path_key, ""))) for item in payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files_json", type=Path)
    parser.add_argument("--regex", required=True)
    parser.add_argument("--path-key", default="filename")
    args = parser.parse_args()

    try:
        payload: object = json.loads(args.files_json.read_text())
        matches = matches_file_payload(payload, args.regex, args.path_key)
    except (OSError, json.JSONDecodeError, TypeError, re.error) as error:
        print(f"{args.files_json}: {error}", file=sys.stderr)
        return 2
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
