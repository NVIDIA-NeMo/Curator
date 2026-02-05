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

"""
Get the latest NeMo Curator Docker image tag from NGC.

Usage:
    python get_latest_tag.py           # Returns latest tag (e.g., "25.09")
    python get_latest_tag.py --full    # Returns full image path
    python get_latest_tag.py --json    # Returns JSON with all info
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error

NGC_IMAGE = "nvcr.io/nvidia/nemo-curator"
NGC_API_URL = "https://api.ngc.nvidia.com/v2/repos/nvidia/nemo-curator"

# Fallback version if API is unavailable
FALLBACK_VERSION = "25.09"


def get_tags_from_api() -> list[str]:
    """Fetch available tags from NGC API."""
    try:
        req = urllib.request.Request(
            f"{NGC_API_URL}",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            # NGC API returns tags in the repository info
            if "tags" in data:
                return data["tags"]
            # Try alternate structure
            if "repository" in data and "tags" in data["repository"]:
                return data["repository"]["tags"]
    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        pass
    return []


def get_tags_from_docker() -> list[str]:
    """Try to get tags using docker CLI (if available)."""
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "manifest", "inspect", NGC_IMAGE, "--verbose"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # This would need parsing, but manifest inspect doesn't list all tags
        # Fall back to skopeo if available
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    try:
        # Try skopeo if available (common on Linux)
        result = subprocess.run(
            ["skopeo", "list-tags", f"docker://{NGC_IMAGE}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("Tags", [])
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    return []


def parse_version_tag(tag: str) -> tuple[int, int] | None:
    """Parse a YY.MM version tag, returns (year, month) or None."""
    match = re.match(r"^(\d{2})\.(\d{2})$", tag)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def get_latest_version_tag(tags: list[str]) -> str:
    """Find the latest YY.MM version tag from a list of tags."""
    version_tags = []
    for tag in tags:
        parsed = parse_version_tag(tag)
        if parsed:
            version_tags.append((parsed, tag))

    if not version_tags:
        return FALLBACK_VERSION

    # Sort by (year, month) descending
    version_tags.sort(key=lambda x: x[0], reverse=True)
    return version_tags[0][1]


def get_latest_tag() -> dict:
    """Get the latest NeMo Curator tag with metadata."""
    # Try API first
    tags = get_tags_from_api()

    # Fall back to docker/skopeo
    if not tags:
        tags = get_tags_from_docker()

    # Determine latest
    if tags:
        latest = get_latest_version_tag(tags)
        source = "ngc_api" if get_tags_from_api() else "docker"
    else:
        latest = FALLBACK_VERSION
        source = "fallback"

    return {
        "tag": latest,
        "image": f"{NGC_IMAGE}:{latest}",
        "source": source,
        "available_tags": sorted(tags, reverse=True)[:10] if tags else [FALLBACK_VERSION],
    }


def main():
    parser = argparse.ArgumentParser(description="Get latest NeMo Curator Docker tag")
    parser.add_argument("--full", action="store_true", help="Print full image path")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    result = get_latest_tag()

    if args.json:
        print(json.dumps(result, indent=2))
    elif args.full:
        print(result["image"])
    else:
        print(result["tag"])


if __name__ == "__main__":
    main()
