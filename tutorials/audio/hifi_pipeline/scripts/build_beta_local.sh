#!/bin/bash
# Local build flow for curator-hifi-beta.
#
# 1. docker build -> 2. enroot import (-> .sqsh on local disk) -> 3. scp to cluster.
#
# This is the alternative to scripts/create_sqsh.sh, which expects the
# image to already be on NGC and runs `enroot import` on the cluster
# login node.  Use this script when you want to iterate on Dockerfile.beta
# without pushing to NGC every time.
#
# Smoke test: after `docker build` we run `docker run --rm <image> --help`
# to confirm Python imports resolve and argparse works.  No GPU required.
#
# Usage:
#   bash tutorials/audio/hifi_pipeline/scripts/build_beta_local.sh
#
# Required env (override defaults via export):
#   CLUSTER_HOST   - SSH alias / host of draco-oci (e.g., draco-oci-iad)
#   CLUSTER_SQSH   - destination dir on cluster (e.g., /lustre/fsw/.../containers/)
#   LOCAL_SQSH_DIR - where to drop the .sqsh locally before scp (default: /tmp)

set -euo pipefail

CURATOR_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${CURATOR_ROOT}"

IMAGE_NAME="${IMAGE_NAME:-curator-hifi-beta}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LOCAL_SQSH_DIR="${LOCAL_SQSH_DIR:-/tmp}"
SQSH_FILE="${LOCAL_SQSH_DIR}/${IMAGE_NAME}.sqsh"

CLUSTER_HOST="${CLUSTER_HOST:-}"
CLUSTER_SQSH="${CLUSTER_SQSH:-}"

DOCKERFILE="tutorials/audio/hifi_pipeline/Dockerfile.beta"

# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------
echo "[1/4] docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f ${DOCKERFILE} ."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${DOCKERFILE}" .

# ---------------------------------------------------------------------------
# 2. Smoke test — image starts, run_pipeline_beta.py imports cleanly,
#    argparse renders --help.  Keeps cycles tight before we burn on enroot.
# ---------------------------------------------------------------------------
echo "[2/4] smoke: docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} --help"
if ! docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" --help > /tmp/beta_help.out 2>&1; then
    echo "ERROR: smoke test failed.  Output:"
    cat /tmp/beta_help.out
    exit 1
fi
head -20 /tmp/beta_help.out
echo "    smoke OK."
echo

# ---------------------------------------------------------------------------
# 3. enroot import -> .sqsh
# ---------------------------------------------------------------------------
if ! command -v enroot >/dev/null 2>&1; then
    echo "ERROR: enroot not installed locally.  Install enroot or skip this step"
    echo "       and use scripts/create_sqsh.sh on the cluster after pushing to NGC."
    exit 1
fi

if [[ -f "${SQSH_FILE}" ]]; then
    echo "[3/4] removing stale ${SQSH_FILE}"
    rm -f "${SQSH_FILE}"
fi

echo "[3/4] enroot import -> ${SQSH_FILE}"
ENROOT_SQUASH_OPTIONS="-comp gzip" enroot import \
    --output "${SQSH_FILE}" \
    "dockerd://${IMAGE_NAME}:${IMAGE_TAG}"
ls -lh "${SQSH_FILE}"
echo

# ---------------------------------------------------------------------------
# 4. scp to cluster (optional)
# ---------------------------------------------------------------------------
if [[ -z "${CLUSTER_HOST}" || -z "${CLUSTER_SQSH}" ]]; then
    echo "[4/4] CLUSTER_HOST / CLUSTER_SQSH not set, skipping scp."
    echo "      To upload manually:"
    echo "        scp ${SQSH_FILE} <user>@<draco-oci-host>:<containers-dir>/"
    exit 0
fi

echo "[4/4] scp ${SQSH_FILE} -> ${CLUSTER_HOST}:${CLUSTER_SQSH}/"
scp "${SQSH_FILE}" "${CLUSTER_HOST}:${CLUSTER_SQSH}/"
echo "    done."

echo
echo "Next: ssh ${CLUSTER_HOST} and run"
echo "  bash tutorials/audio/hifi_pipeline/submit_beta.sh --corpus ytc_ru --dry-run"
