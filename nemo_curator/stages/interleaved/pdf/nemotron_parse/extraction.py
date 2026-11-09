# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the installed served parser's artifact identity without model loading."""

import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path


def parser_package_identity() -> dict[str, str]:
    """Bind the distribution version and ordered Python source inventory."""
    spec = importlib.util.find_spec("nemotron_parse")
    if spec is None or spec.origin is None:
        msg = "The nemotron-parse package is required for page-budget extraction identity"
        raise ValueError(msg)
    root = Path(spec.origin).parent
    digest = hashlib.sha256()
    sources = sorted(root.rglob("*.py"))
    if not sources:
        msg = "Cannot attest nemotron-parse: its Python source inventory is empty"
        raise ValueError(msg)
    for path in sources:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"version": importlib.metadata.version("nemotron-parse"), "source_sha256": digest.hexdigest()}
