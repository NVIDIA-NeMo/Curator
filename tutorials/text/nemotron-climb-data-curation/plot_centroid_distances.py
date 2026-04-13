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

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main(args: argparse.Namespace) -> None:
    # Load centroids and compute pairwise distances
    centroids = np.load(os.path.join(args.centroids_path, "kmeans_centroids.npy"))
    diff = centroids[:, None, :] - centroids[None, :, :]
    l2_distances = np.sqrt(np.sum(diff**2, axis=2))

    # Ignore diagonal (distance to self)
    dists = l2_distances[np.triu_indices_from(l2_distances, k=1)]

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(dists, bins=50, color="skyblue", edgecolor="black")
    plt.title("Histogram of Pairwise Centroid L2 Distances")
    plt.xlabel("L2 distance")
    plt.ylabel("Number of centroid pairs")

    # Save image
    plt.savefig(args.image_path, dpi=300)
    plt.close()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--centroids-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
