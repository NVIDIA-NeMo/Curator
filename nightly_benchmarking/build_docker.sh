#!/usr/bin/env bash
set -euo pipefail

# Build the benchmarking image using ray-curator/Dockerfile
# Usage:
#   micromamba run -n ray_curator_2506 bash nightly_benchmarking/build_docker.sh [tag]

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-curator-bench:$(date +%Y%m%d)-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"

echo "Building Docker image: ${TAG}"
echo "Repo root: ${REPO_ROOT}"

export DOCKER_BUILDKIT=1

micromamba run -n ray_curator_2506 docker build \
  -f "$REPO_ROOT/ray-curator/Dockerfile" \
  -t "$TAG" \
  --build-arg CURATOR_ENV=dev \
  "$REPO_ROOT"

# Print image digest if available
if command -v docker >/dev/null 2>&1; then
  DIGEST=$(micromamba run -n ray_curator_2506 docker inspect --format='{{index .RepoDigests 0}}' "$TAG" || true)
  echo "Built image: $TAG  digest: ${DIGEST:-unknown}"
fi

echo "Done."
