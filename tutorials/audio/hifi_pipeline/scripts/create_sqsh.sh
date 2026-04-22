#!/bin/bash
# Create .sqsh container images from nvcr.io for each HIFI pipeline stage.
#
# Run on cluster login node:
#   bash create_sqsh.sh
#
# Requires: enroot, NGC credentials (~/.config/enroot/.credentials)

set -euo pipefail

DEST="${SQSH_DIR:-/path/to/containers}"
NGC_ORG="${NGC_ORG:-nvidian/ac-aiapps}"
mkdir -p "${DEST}"

IMAGES=(
    "curator-utmos:latest"
    "curator-hifi-nemo-stages:latest"
    "curator-hifi-pipeline:latest"
)

for img in "${IMAGES[@]}"; do
    NAME="${img%%:*}"
    TAG="${img##*:}"
    SQSH="${DEST}/${NAME}.sqsh"

    if [ -f "${SQSH}" ]; then
        echo "=== ${NAME}: already exists at ${SQSH}, skipping ==="
        echo "    (delete it first to rebuild)"
        continue
    fi

    echo "=== ${NAME}: importing from nvcr.io/${NGC_ORG}/${img} ==="
    # gzip compression: default is -noD (uncompressed data), which produces huge sqsh files
    ENROOT_SQUASH_OPTIONS="-comp gzip" enroot import --output "${SQSH}" "docker://nvcr.io/${NGC_ORG}/${img}"
    echo "    -> ${SQSH} ($(du -h "${SQSH}" | cut -f1))"
    echo
done

echo "Done. Container images:"
ls -lh "${DEST}"/curator-*.sqsh
