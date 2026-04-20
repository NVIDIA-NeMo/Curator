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
# https://github.com/sail-sg/regmix/blob/main/regression_fitting/regression.ipynb

"""
For predictor training, we use a LightGBM regression model, which fits mixture-performance pairs well with limited data.
To prevent overfitting, we set L1 and L2 regularization, early stopping, a maximum depth of four, and require at least five samples per leaf.
Additionally, we employed a separate validation set and an early stopping mechanism, halting training after 20 rounds of no improvement.
"""

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils import get_token_distribution

SEED = 42
np.random.seed(SEED)  # noqa: NPY002


def load_benchmark_results(
    input_path: str, domain_names: list[str], mixtures_path: str, output_path: str
) -> pd.DataFrame:
    input_path = Path(input_path)

    # Initialize an empty DataFrame
    df = pd.DataFrame(
        columns=[
            "model_benchmark_path",
            "arc_easy_acc",
            "piqa_acc_norm",
            "hellaswag_acc_norm",
            "valid_avg",
            *domain_names,
        ]
    )

    # Construct the DataFrame of benchmark results and data mixtures per model
    for model_dirs in sorted(input_path.iterdir(), key=lambda p: p.name):
        # Initialize a Series with the same columns as the DataFrame
        series = pd.Series(index=df.columns)
        # Grab model name from model directory name, e.g., n1, n2, etc.
        model_name = model_dirs.name

        # Grab relevant subdirectory from each model directory
        subdirs = [d for d in model_dirs.iterdir() if d.is_dir()]
        assert len(subdirs) == 1, "Expected exactly one subdirectory per model directory"  # noqa: S101
        model_dir = subdirs[0]

        # Grab benchmark result file for the model
        jsons = sorted(model_dir.glob("results_*.json"))
        if not jsons:
            msg = f"No benchmark result file found for the model {model_dir}. Check if the input path is correct."
            raise RuntimeError(msg)

        # Grab the benchmark results
        data = json.loads(jsons[-1].read_text()).get("results")
        arc_easy_acc = data.get("arc_easy").get("acc,none") * 100
        piqa_acc_norm = data.get("piqa").get("acc_norm,none") * 100
        hellaswag_acc_norm = data.get("hellaswag").get("acc_norm,none") * 100
        valid_avg = (arc_easy_acc + piqa_acc_norm + hellaswag_acc_norm) / 3

        # Assign the values to the Series
        series["model_benchmark_path"] = model_dirs
        series["arc_easy_acc"] = arc_easy_acc
        series["piqa_acc_norm"] = piqa_acc_norm
        series["hellaswag_acc_norm"] = hellaswag_acc_norm
        series["valid_avg"] = valid_avg

        # Grab the data mixture for the model
        data_mixture = os.path.join(mixtures_path, f"{model_name}.sh")

        # Grab the corresponding data mixture for the model
        with open(data_mixture) as f:
            # Ignore the first line and any line containing "EOF"
            for line in f:
                if line.startswith("#") or line.startswith("cat") or line.startswith("EOF"):  # noqa: PIE810
                    continue
                weight, domain_name = line.strip().split()
                domain_name = domain_name.split("/")[-1]
                assert domain_name in domain_names, f"Domain {domain_name} not found in the domains_path"  # noqa: S101
                series[domain_name] = float(weight)

        # Replace NaN with 0
        series = series.fillna(0)

        # Append the Series to the DataFrame
        df = pd.concat([df, series.to_frame().T], ignore_index=True)

    # Save the DataFrame to a CSV file and return it
    df.to_csv(os.path.join(output_path, "lm_harness_results.csv"), index=False)
    return df


def fit_predictor(df: pd.DataFrame, domain_names: list[str], target_column: str) -> lgb.LGBMRegressor:
    # Shuffle the DataFrame
    shuffled_df = df.sample(frac=1, random_state=SEED)

    # Split the DataFrame into train and test sets
    train_df = shuffled_df.iloc[: int(len(shuffled_df) * 0.9)]
    test_df = shuffled_df.iloc[int(len(shuffled_df) * 0.9) :]

    train_df_config = train_df[domain_names]
    train_df_target = train_df[[target_column]]

    test_df_config = test_df[domain_names]
    test_df_target = test_df[[target_column]]

    x_train = train_df_config[train_df_config.columns[0:]].to_numpy()
    y_train = train_df_target[train_df_target.columns[0:]].to_numpy()
    x_test = test_df_config[test_df_config.columns[0:]].to_numpy()
    y_test = test_df_target[test_df_target.columns[0:]].to_numpy()

    # Train the predictor
    hyper_params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": ["l1", "l2"],
        "num_iterations": 1000,
        "seed": 42,
        "learning_rate": 1e-2,
        "verbosity": -1,
    }

    gbm = lgb.LGBMRegressor(**hyper_params)

    return gbm.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=3, verbose=False),
        ],
    )


def generate_mixtures(
    num_mixtures: int, output_path: str, samples: np.ndarray, simulation: np.ndarray, domain_paths: list[str]
) -> None:
    if num_mixtures == 1:
        # Take the average of top-k simulated data mixtures as the optimal data mixture
        k = 128
        top_k_samples = samples[np.argsort(simulation)[0:k]]

        # Get the optimal data mixture by taking the average of top-k samples
        optimal_data_mixture = np.mean(top_k_samples, axis=0)

        # Save n1.sh file
        with open(os.path.join(output_path, "n1.sh"), "w") as f:
            f.write("#!/bin/bash\n")
            f.write("cat <<EOF\n")
            for path, weight in zip(domain_paths, optimal_data_mixture, strict=True):
                if weight > 0:
                    formatted = f"{weight:.4f}".rstrip("0").rstrip(".")
                    if formatted != "0":
                        f.write(f"{formatted} {path}\n")
            f.write("EOF\n")
    else:
        k_pool = num_mixtures * num_mixtures
        top_pool = samples[np.argsort(simulation)[:k_pool]]

        # Cluster the top-k samples into diverse mixtures
        km = KMeans(n_clusters=num_mixtures, random_state=SEED)
        km.fit(top_pool)
        diverse_mixtures = km.cluster_centers_
        # Renormalize since centroids may not sum to exactly 1
        diverse_mixtures /= diverse_mixtures.sum(axis=1, keepdims=True)

        # Save n1.sh, ..., n{args.num_mixtures}.sh files for each diverse mixture
        for i in range(num_mixtures):
            with open(os.path.join(output_path, f"n{i + 1}.sh"), "w") as f:
                f.write("#!/bin/bash\n")
                f.write("cat <<EOF\n")
                for path, weight in zip(domain_paths, diverse_mixtures[i], strict=True):
                    if weight > 0:
                        formatted = f"{weight:.4f}".rstrip("0").rstrip(".")
                        if formatted != "0":
                            f.write(f"{formatted} {path}\n")
                f.write("EOF\n")


def main(args: argparse.Namespace) -> None:
    # Initialize the output path if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Grab all the bin files under domains_path
    domains_path = Path(args.domains_path)
    bin_files = sorted(domains_path.glob("*.bin"))
    # Grab file names without extension (these are the domain names)
    domain_names = [file.stem for file in bin_files]

    if args.input_path is not None and args.mixtures_path is not None:
        df = load_benchmark_results(
            input_path=args.input_path,
            domain_names=domain_names,
            mixtures_path=args.mixtures_path,
            output_path=args.output_path,
        )
    elif args.lm_harness_results_csv_path is not None:
        df = pd.read_csv(args.lm_harness_results_csv_path)
    else:
        msg = "Either (--input-path, --mixtures-path) or (--lm-harness-results-csv-path) must be provided"
        raise ValueError(msg)

    predictor = fit_predictor(df, domain_names, args.metric)

    # Get the token distribution of each domain
    token_dist = get_token_distribution(args.domains_path)
    prior_dist = [token_dist[str(f)] for f in bin_files]
    domain_paths = [str(f.with_suffix("")) for f in bin_files]

    samples = np.random.dirichlet(prior_dist * 1, 100000)  # noqa: NPY002
    simulation = predictor.predict(samples)

    generate_mixtures(args.num_mixtures, args.output_path, samples, simulation, domain_paths)


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # I/O args
    parser.add_argument("--input-path", type=str, required=False)
    parser.add_argument("--lm-harness-results-csv-path", type=str, required=False)
    parser.add_argument("--domains-path", type=str, required=True)
    parser.add_argument("--mixtures-path", type=str, required=False)
    parser.add_argument("--output-path", type=str, required=True)

    # Prediction args
    parser.add_argument("--metric", type=str, default="valid_avg")
    parser.add_argument("--num-mixtures", type=int, default=1)

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
