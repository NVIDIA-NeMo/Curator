#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# Pynini bundles OpenFst in x86 wheels, but requires native libraries on ARM.
set -euo pipefail
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != --check ]; }; then
    echo "Usage: install_openfst.sh [--check]" >&2
    exit 2
fi
if [ "$(uname -m)" != aarch64 ]; then
    exit 0
fi

check_openfst() {
    [ -f /usr/local/include/fst/fstlib.h ] || return 1
    local library
    for library in fst fstscript fstfar fstfarscript fstpdtscript fstmpdtscript; do
        [ -f "/usr/local/lib/lib${library}.so" ] || return 1
    done
}

if check_openfst; then
    exit 0
fi
if [ "${1:-}" = --check ]; then
    echo "ERROR: OpenFst libraries required to build Pynini on ARM are missing." >&2
    exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must be run as root to install OpenFst." >&2
    exit 1
fi

apt-get update
apt-get install -y --no-install-recommends build-essential ca-certificates curl
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl -fLsS https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.3.tar.gz -o "$work/openfst.tar.gz"
echo "077714159d5cf3e38a80b6c6656d3ccc2c8b8b6c50bb41bb65c5fec10796bf53  $work/openfst.tar.gz" | sha256sum -c -
tar -xzf "$work/openfst.tar.gz" -C "$work"
cd "$work/openfst-1.8.3"
./configure --prefix=/usr/local --enable-grm --disable-static CXXFLAGS='-O2 -g0'
# Keep script libraries enabled, but avoid compiling the unused CLI programs.
make -C src/lib -j4
make -C src/script -j4
make -C src/extensions/far -j4 libfstfar.la libfstfarscript.la
make -C src/extensions/pdt -j4 libfstpdtscript.la
make -C src/extensions/mpdt -j4 libfstmpdtscript.la
make -C src/include install
make -C src/lib install
make -C src/script install
for extension in far pdt mpdt; do
    make -C "src/extensions/$extension" install-libLTLIBRARIES
done
ldconfig
check_openfst
