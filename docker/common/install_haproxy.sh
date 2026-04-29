#!/bin/bash
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

set -xeuo pipefail

# Build HAProxy from source so Ray Serve's HAProxy ingress mode (RAY_SERVE_ENABLE_HA_PROXY=1)
# has the binary on $PATH. Mirrors ray-project/ray docker/base-slim/Dockerfile (Ray 2.55+).
#
# Fetched from ray-project/haproxy-release (a GitHub release mirror) because
# www.haproxy.org's wildcard TLS cert expired 2026-04-17 and the release tarball
# disappeared from the upstream download site. Switch back to www.haproxy.org once
# the cert is renewed and the tarball is republished upstream.
HAPROXY_VERSION=2.8.20
HAPROXY_SHA256=c8301de11dabfbf049db07080e43b9570a63f99e41d4b0754760656bf7ea00b7

for i in "$@"; do
    case $i in
        --HAPROXY_VERSION=?*) HAPROXY_VERSION="${i#*=}";;
        --HAPROXY_SHA256=?*) HAPROXY_SHA256="${i#*=}";;
        *) ;;
    esac
done

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libc6-dev \
    liblua5.3-dev \
    libpcre3-dev \
    libssl-dev \
    zlib1g-dev \
    liblua5.3-0 \
    socat
rm -rf /var/lib/apt/lists/*

BUILD_DIR=$(mktemp -d)
curl --retry 5 --retry-all-errors --connect-timeout 20 --max-time 300 \
     -fsSL -o "${BUILD_DIR}/haproxy.tar.gz" \
     "https://github.com/ray-project/haproxy-release/releases/download/${HAPROXY_VERSION}/haproxy-${HAPROXY_VERSION}.tar.gz"
echo "${HAPROXY_SHA256}  ${BUILD_DIR}/haproxy.tar.gz" | sha256sum -c -
tar -xzf "${BUILD_DIR}/haproxy.tar.gz" -C "${BUILD_DIR}" --strip-components=1
make -C "${BUILD_DIR}" TARGET=linux-glibc \
    USE_OPENSSL=1 USE_ZLIB=1 USE_PCRE=1 USE_LUA=1 USE_PROMEX=1 -j"$(nproc)"
make -C "${BUILD_DIR}" install SBINDIR=/usr/local/bin
rm -rf "${BUILD_DIR}"

# Drop build-only ``-dev`` deps so they don't bloat the image. Do NOT pass
# ``--auto-remove`` here — it would also reap the runtime libs (libpcre3,
# libssl3, zlib1g, libc6) that HAProxy linked against, breaking ``haproxy -v``
# with ``libpcre.so.3: cannot open shared object file``.
apt-get purge -y \
    build-essential \
    libc6-dev \
    liblua5.3-dev \
    libpcre3-dev \
    libssl-dev \
    zlib1g-dev
rm -rf /var/lib/apt/lists/*

mkdir -p /etc/haproxy /run/haproxy /var/log/haproxy

haproxy -v
