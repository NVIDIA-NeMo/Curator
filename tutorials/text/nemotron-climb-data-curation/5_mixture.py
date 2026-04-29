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

# This script is adapted from the RegMix project:
# https://github.com/sail-sg/regmix/blob/main/mixture_config/synthesize_mixture.py

import argparse
import os
import random

import numpy as np
from utils import get_token_distribution

SEED = 42
random.seed(SEED)
np.random.seed(SEED)  # noqa: NPY002


def generate_train_group(groups: list[str], weights: list[float], precision: int = 5) -> dict[str, float]:
    """
    Generate a dictionary of groups and their corresponding weights.

    Args:
    groups (list): List of group names.
    weights (list): List of corresponding weights.
    precision (int): Number of decimal places for rounding weights. Defaults to 5.

    Returns:
    dict: Dictionary of groups and their corresponding weights.
    """

    # Remove ".bin" from the end of the group names
    groups = [group.removesuffix(".bin") for group in groups]

    # Round the weights to the precision
    weights = [round(weight, precision) for weight in weights]

    return dict(zip(groups, weights, strict=True))


def generate_weights_dirichlet(  # noqa: C901, PLR0912, PLR0913
    prior_dist: list[float],
    train_groups: list[str],
    minimum_number: float,
    num_samples: int = 128,
    enable_bound: bool = True,
    temperature: float = 0.5,
    maximum_usage: int = 15,
    sample_multiplier: int = 100,
    min_strength: float = 0.1,
    max_strength: float = 5.0,
) -> np.ndarray:
    """
    Generate the weights for the train groups using the Dirichlet distribution.

    Args:
    prior_dist (list): List of prior distribution weights.
    train_groups (list): List of train group names.
    minimum_number (float): Minimum number of samples for each group.
    num_samples (int): Number of samples to generate. Defaults to 128.
    enable_bound (bool): Whether to enable the bound for reject sampling. Defaults to True.
    temperature (float): Temperature for the Dirichlet distribution. Defaults to 0.5.
    maximum_usage (int): Maximum usage for each group. Defaults to 15.
    sample_multiplier (int): Sample multiplier for the Dirichlet distribution. Defaults to 100.
    min_strength (float): Minimum strength for the Dirichlet distribution. Defaults to 0.1.
    max_strength (float): Maximum strength for the Dirichlet distribution. Defaults to 5.0.

    Returns:
    np.ndarray: Array of weights for the train groups.
    """

    final_samples = []

    if enable_bound:
        # generate the bound for reject sampling
        number_bound = []
        for i in range(len(prior_dist)):
            # the token cannot be used more than 4 times
            number_bound.append([0.0, min(prior_dist[i] * maximum_usage, 1.0)])
    else:
        number_bound = None

    # apply temperature
    if temperature < 1.0:
        prior_dist = prior_dist**temperature
        prior_dist = prior_dist / np.sum(prior_dist)
        print("\n\nWith temperature: ", prior_dist)

    print("\n\nThe domain usage bound (maximum domain weight): ")
    # print the bound for each group
    for i in range(len(prior_dist)):
        print(f"{train_groups[i]}: {number_bound[i][1]}")

    # combine reject sampling with dirichlet distribution
    for _ in range(num_samples * sample_multiplier):
        if min_strength == max_strength:
            samples = np.random.dirichlet(prior_dist * min_strength, 1)  # noqa: NPY002
        else:
            samples = []
            min_strength_log = np.log10(min_strength)
            max_strength_log = np.log10(max_strength)
            for strength in np.logspace(min_strength_log, max_strength_log, 15):
                # add a noise to the strength
                samples_per_strength = np.random.dirichlet(prior_dist * strength, 1)  # noqa: NPY002
                samples.append(samples_per_strength)
            # random sample one
            samples = random.choice(samples)  # noqa: S311

        # if there is a bound, the bound is a list of tuples indicating the lower and upper bound of each group
        ensure_flag = True
        if number_bound is not None:
            for j in range(len(samples[0])):
                if samples[0][j] < number_bound[j][0] or samples[0][j] > number_bound[j][1]:
                    ensure_flag = False
                    break

        if ensure_flag is False:
            continue

        # post normalization, set zero for the number less than minimum_number
        samples = np.where(samples < minimum_number, 0.0, samples)
        # round samples into the same scale of minimum_number
        samples = samples / np.sum(samples, axis=1).reshape(-1, 1)
        samples = np.round(samples / minimum_number) * minimum_number
        # add the samples to the final_samples
        final_samples.append(samples[0])

    # remove the samples with the nearly same values
    print("\nThe number of available samples: ", len(final_samples))
    final_samples = sort_and_deduplicate(np.array(final_samples))
    print("The number of deduplicated samples: ", len(final_samples))
    selected_samples = random.sample(final_samples, num_samples)
    print("The number of selected samples: ", len(selected_samples))
    return np.stack(selected_samples, axis=0)


def generate_config_from_prior(  # noqa: PLR0913
    output_paths: list[str],
    prior_config: dict[str, dict[str, float]],
    temp: float,
    min_strength: float,
    max_strength: float,
    sample_multiplier: int,
    maximum_usage: int,
    minimum: float,
) -> None:
    """
    Generate the config from the prior distribution.

    Args:
    output_paths (list): List of output paths.
    prior_config (dict): Dictionary of prior distribution.
    temp (float): Temperature for the Dirichlet distribution.
    min_strength (float): Minimum strength for the Dirichlet distribution.
    max_strength (float): Maximum strength for the Dirichlet distribution.
    sample_multiplier (int): Sample multiplier for the Dirichlet distribution.
    maximum_usage (int): Maximum usage for each group.
    minimum (float): Minimum number of samples for each group.

    Returns:
    None (mixture config files are written to the output paths).
    """

    number_of_samples = len(output_paths)
    train_config = prior_config
    train_groups, prior_dist = [], []

    for k, v in train_config.items():
        train_groups.append(k)
        prior_dist.append(v)

    # renormalize the prior distribution
    prior_dist = prior_dist / np.sum(prior_dist)
    print("Prior distribution after normalization: ", prior_dist)

    train_weights = generate_weights_dirichlet(
        prior_dist=prior_dist,
        train_groups=train_groups,
        minimum_number=minimum,
        num_samples=number_of_samples,
        temperature=temp,
        maximum_usage=maximum_usage,
        sample_multiplier=sample_multiplier,
        min_strength=min_strength,
        max_strength=max_strength,
    )

    for output_path, weights in zip(output_paths, train_weights, strict=True):
        # get the train group
        train_group = generate_train_group(train_groups, weights)

        with open(output_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("cat <<EOF\n")
            for path, weight in train_group.items():
                if weight > 0:
                    f.write(f"{weight} {path}\n")
            f.write("EOF\n")


def sort_and_deduplicate(data: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """
    Remove identify configs to avoid duplicated training.

    Args:
    data (np.ndarray): Array of weights for the train groups.
    threshold (float): Threshold for deduplication. Defaults to 1e-5.

    Returns:
    np.ndarray: Array of deduplicated weights for the train groups.
    """

    arr = np.array(data)
    sorted_indices = np.lexsort(arr.T)
    sorted_arr = arr[sorted_indices]
    result = [sorted_arr[0]]

    for i in range(1, len(sorted_arr)):
        diff = np.sum(np.abs(sorted_arr[i] - result[-1]))
        if diff > threshold:
            result.append(sorted_arr[i])

    return result


def main(args: argparse.Namespace) -> None:
    input_path = args.input_path
    output_path = args.output_path
    num_mixtures = args.num_mixtures

    # if not exist, create the folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_paths = []
    for i in range(1, num_mixtures + 1):
        output_paths.append(f"{output_path}/n{i}.sh")

    generate_config_from_prior(
        output_paths,
        prior_config=get_token_distribution(input_path),
        temp=args.temp,
        min_strength=args.min_strength,
        max_strength=args.max_strength,
        sample_multiplier=args.sample_multiplier,
        maximum_usage=args.maximum_usage,
        minimum=args.minimum,
    )


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # I/O args
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-mixtures", type=int, required=True)

    # Temperature for the prior distribution, if your distribution is too skewed, you can use a temperature to smooth it
    parser.add_argument("--temp", type=float, default=0.5)

    # The minimum and maximum strength for the dirichlet distribution.
    # With a small value, the distribution will be more concentrated, and with a large value, the distribution will be more uniform.
    parser.add_argument("--min-strength", type=float, default=0.1)
    parser.add_argument("--max-strength", type=float, default=5.0)

    # We first sample SAMPLE_MULTIPLIER times more samples than randomly select some of them
    parser.add_argument("--sample-multiplier", type=int, default=100)

    # How many epochs are allowed for each domain for the large-scale model training. This hyper-parameter
    #   is used because the natura trade off between the reweighting v.s. the number of avaiable tokens in each domain.
    #   Usually we think repeating 4 epochs is okay for language model pre-training, and here we set it as 15
    #   because the avaiable token of The Pile is much larger than the token amount for training Chinchilla-Optimal 1B models (i.e., 25B tokens).
    #   However, if you want to train the large-scale model with all avaiable tokens, you can use less than 4 epochs also in the proxy
    #   model training.
    parser.add_argument("--maximum-usage", type=int, default=15)

    # Assume that we have 1B (512,000 examples, and 2048 tokens per example) tokens
    #   for the proxy model training, the minimum sampling rate 2e-4 indicates that
    #   at least there will be 100 examples for each domain, which is statistically significant.
    #
    # If you use less tokens for training the proxy models, you may increase the minimum sampling rate
    #   to ensure the statistical significance of the domain. I personally recommend using at least 1e-5
    #   if you have 1B tokens for training the proxy models.
    parser.add_argument("--minimum", type=float, default=2e-4)

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
