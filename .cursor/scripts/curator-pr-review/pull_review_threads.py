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
"""Pull every review thread and every comment in each thread with GraphQL cursors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

THREADS_QUERY = """
query($owner:String!,$repo:String!,$pr:Int!,$cursor:String) {
  repository(owner:$owner,name:$repo) {
    pullRequest(number:$pr) {
      reviewThreads(first:100,after:$cursor) {
        nodes {
          id isResolved isOutdated isCollapsed line originalLine path
          comments(first:100) {
            nodes { databaseId body }
            pageInfo { hasNextPage endCursor }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""

COMMENTS_QUERY = """
query($threadId:ID!,$cursor:String) {
  node(id:$threadId) {
    ... on PullRequestReviewThread {
      comments(first:100,after:$cursor) {
        nodes { databaseId body }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


def graphql(query: str, variables: dict[str, object]) -> dict[str, Any]:
    command = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        if value is None:
            continue
        flag = "-F" if isinstance(value, int) else "-f"
        command.extend([flag, f"{key}={value}"])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    payload: dict[str, Any] = json.loads(completed.stdout)
    if payload.get("errors"):
        sys.stderr.write(json.dumps(payload["errors"], indent=2) + "\n")
        raise SystemExit(1)
    return payload


def connection_page(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        current = current[key]
    if not isinstance(current, dict):
        msg = f"GraphQL path {'.'.join(keys)} is not an object"
        raise TypeError(msg)
    return current


def append_remaining_comments(thread: dict[str, Any]) -> None:
    connection = thread["comments"]
    seen_cursors: set[str] = set()
    while connection["pageInfo"]["hasNextPage"]:
        cursor = connection["pageInfo"]["endCursor"]
        if not cursor or cursor in seen_cursors:
            msg = f"comment pagination cursor did not advance for thread {thread['id']}"
            raise RuntimeError(msg)
        seen_cursors.add(cursor)
        payload = graphql(COMMENTS_QUERY, {"threadId": thread["id"], "cursor": cursor})
        connection = connection_page(payload, "data", "node", "comments")
        thread["comments"]["nodes"].extend(connection["nodes"])
        thread["comments"]["pageInfo"] = connection["pageInfo"]


def pull_all(owner: str, repo: str, pr: int) -> dict[str, Any]:
    threads: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()

    while True:
        payload = graphql(THREADS_QUERY, {"owner": owner, "repo": repo, "pr": pr, "cursor": cursor})
        try:
            connection = connection_page(payload, "data", "repository", "pullRequest", "reviewThreads")
        except (KeyError, TypeError) as error:
            msg = f"PR {owner}/{repo}#{pr} was not returned by GraphQL"
            raise RuntimeError(msg) from error

        for thread in connection["nodes"]:
            append_remaining_comments(thread)
            threads.append(thread)

        page_info = connection["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
        if not cursor or cursor in seen_cursors:
            msg = "review-thread pagination cursor did not advance"
            raise RuntimeError(msg)
        seen_cursors.add(cursor)

    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": threads,
                        "pageInfo": {"hasNextPage": False, "endCursor": cursor},
                    }
                }
            }
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="OWNER/REPOSITORY")
    parser.add_argument("--pr", required=True, type=int)
    args = parser.parse_args()
    if "/" not in args.repo:
        parser.error("--repo must use OWNER/REPOSITORY format")
    owner, repo = args.repo.split("/", 1)
    json.dump(pull_all(owner, repo, args.pr), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
