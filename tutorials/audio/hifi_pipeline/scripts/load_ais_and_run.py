#!/usr/bin/env python3
"""Read AIS_AUTHN_TOKEN from .ais_token file into os.environ, then exec target script.

Workaround for an env-propagation bug seen on Draco-OCI with the
curator-hifi-pipeline.sqsh container: bash 'export AIS_AUTHN_TOKEN=...'
inside the container does not always carry the freshly-sourced value
to a child Python's os.environ — Python sees a stale baked token
(length 183, missing 'sub' claim) instead of the user's real token
(length 228).  Re-reading the token file directly inside Python
bypasses the broken chain.

Usage:
    python load_ais_and_run.py /path/to/run_pipeline.py [args ...]

Sanity: confirmed by job 9451754 — driver=228, actor=228, tar reads
return valid bytes.
"""
from __future__ import annotations

import os
import runpy
import sys


def _load_ais_token() -> None:
    candidates = [
        "/root/.ais_token",
        "/home/gzelenfroind/.ais_token",
        os.path.expanduser("~/.ais_token"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if "AIS_AUTHN_TOKEN" in line and "=" in line:
                        value = line.split("=", 1)[1].strip().strip("'\"")
                        if value:
                            os.environ["AIS_AUTHN_TOKEN"] = value
                            return
        except OSError:
            continue


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: load_ais_and_run.py <target.py> [args ...]", file=sys.stderr)
        sys.exit(2)

    _load_ais_token()

    target = sys.argv[1]
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
