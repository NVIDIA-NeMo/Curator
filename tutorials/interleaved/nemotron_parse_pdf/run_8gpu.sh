#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/direct_launcher.sh"

run_direct_profile "" nemotron-parse-pdf-direct-r16-w36
