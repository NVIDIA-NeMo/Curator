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

# Versions pinned to match upstream ai-dynamo/dynamo container/context.yaml,
# so Dynamo-Curator integration runs the same etcd/nats-server binaries as
# Dynamo's own runtime images.
ETCD_VERSION=3.5.21
NATS_VERSION=2.10.28

for i in "$@"; do
    case $i in
        --ETCD_VERSION=?*) ETCD_VERSION="${i#*=}";;
        --NATS_VERSION=?*) NATS_VERSION="${i#*=}";;
        *) ;;
    esac
    shift
done

ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  ETCD_ARCH="amd64"; NATS_ARCH="amd64" ;;
    aarch64) ETCD_ARCH="arm64"; NATS_ARCH="arm64" ;;
    *) echo "Unsupported architecture: $ARCH" && exit 1 ;;
esac

echo "Installing etcd ${ETCD_VERSION} (${ETCD_ARCH})..."
curl -fsSL -o /tmp/etcd.tar.gz \
    "https://github.com/etcd-io/etcd/releases/download/v${ETCD_VERSION}/etcd-v${ETCD_VERSION}-linux-${ETCD_ARCH}.tar.gz"
tar xzf /tmp/etcd.tar.gz -C /tmp/
install -m 0755 /tmp/etcd-v${ETCD_VERSION}-linux-${ETCD_ARCH}/etcd /usr/local/bin/
install -m 0755 /tmp/etcd-v${ETCD_VERSION}-linux-${ETCD_ARCH}/etcdctl /usr/local/bin/
rm -rf /tmp/etcd*

etcd --version

echo "Installing nats-server ${NATS_VERSION} (${NATS_ARCH})..."
curl -fsSL -o /tmp/nats.tar.gz \
    "https://github.com/nats-io/nats-server/releases/download/v${NATS_VERSION}/nats-server-v${NATS_VERSION}-linux-${NATS_ARCH}.tar.gz"
tar xzf /tmp/nats.tar.gz -C /tmp/
install -m 0755 /tmp/nats-server-v${NATS_VERSION}-linux-${NATS_ARCH}/nats-server /usr/local/bin/
rm -rf /tmp/nats*

nats-server --version

echo "etcd ${ETCD_VERSION} and nats-server ${NATS_VERSION} installed successfully."
