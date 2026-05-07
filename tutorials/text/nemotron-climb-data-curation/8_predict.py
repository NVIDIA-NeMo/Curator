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
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from utils import get_token_distribution

SEED = 42
np.random.seed(SEED)  # noqa: NPY002


def _natural_key(name: str) -> tuple[int | str, ...]:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name))


def load_benchmark_results(
    input_paths: list[str], domain_names: list[str], mixtures_paths: list[str], output_path: str
) -> pd.DataFrame:
    if len(input_paths) != len(mixtures_paths):
        msg = (
            f"Number of --input-paths entries ({len(input_paths)}) must match "
            f"number of --mixtures-paths entries ({len(mixtures_paths)})."
        )
        raise ValueError(msg)

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

    # Construct the DataFrame of benchmark results and data mixtures per model across all input/mixtures pairs
    for input_path_str, mixtures_path in zip(input_paths, mixtures_paths, strict=True):
        input_path = Path(input_path_str)

        for proxy_dir in sorted(input_path.iterdir(), key=lambda p: _natural_key(p.name)):
            # Initialize a Series with the same columns as the DataFrame
            series = pd.Series(index=df.columns)
            # Grab model name from model directory name, e.g., n1, n2, etc.
            model_name = proxy_dir.name

            # Grab relevant subdirectory from each model directory
            subdirs = [d for d in proxy_dir.iterdir() if d.is_dir()]
            assert len(subdirs) == 1, "Expected exactly one subdirectory per model directory"  # noqa: S101
            model_dir = subdirs[0]

            # Grab benchmark result file for the model
            jsons = sorted(model_dir.glob("results_*.json"))
            if not jsons:
                msg = f"No benchmark result file found for the model {model_dir}. Check if the input path is correct."
                raise RuntimeError(msg)

            # Grab the benchmark results
            data = json.loads(jsons[-1].read_text())["results"]
            for bench in ("arc_easy", "piqa", "hellaswag"):
                if bench not in data:
                    msg = f"Benchmark {bench!r} missing from {jsons[-1]}"
                    raise RuntimeError(msg)
            arc_easy_acc = data["arc_easy"]["acc,none"] * 100
            piqa_acc_norm = data["piqa"]["acc_norm,none"] * 100
            hellaswag_acc_norm = data["hellaswag"]["acc_norm,none"] * 100
            valid_avg = (arc_easy_acc + piqa_acc_norm + hellaswag_acc_norm) / 3

            # Assign the values to the Series
            series["model_benchmark_path"] = proxy_dir
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
                    if not line.strip():
                        continue
                    if line.startswith("#") or line.startswith("cat") or line.startswith("EOF"):  # noqa: PIE810
                        continue
                    weight, domain_name = line.strip().split()
                    domain_name = os.path.basename(domain_name)
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
    split_idx = int(len(shuffled_df) * 0.9)
    if split_idx == 0 or split_idx == len(shuffled_df):
        msg = (
            f"fit_predictor needs at least 2 rows to produce non-empty 90/10 train/test splits, "
            f"got {len(shuffled_df)}."
        )
        raise ValueError(msg)
    train_df = shuffled_df.iloc[:split_idx]
    test_df = shuffled_df.iloc[split_idx:]

    # Cap stopping_rounds at len(test_df) so early stopping stays proportional to the eval set
    # Nemotron-CLIMB algorithm starts with 64 proxy model evaluations -> test=7, far below the original stopping_rounds=20
    stopping_rounds = min(20, len(test_df))

    x_train = train_df[domain_names].to_numpy()
    y_train = train_df[target_column].to_numpy()
    x_test = test_df[domain_names].to_numpy()
    y_test = test_df[target_column].to_numpy()

    # Train the predictor
    hyper_params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": ["l1", "l2"],
        "num_iterations": 1000,
        "seed": 42,
        "learning_rate": 1e-2,
        "verbosity": -1,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "max_depth": 4,
        "min_child_samples": 5,
    }

    gbm = lgb.LGBMRegressor(**hyper_params)

    return gbm.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False),
        ],
    )


def generate_mixtures(
    num_mixtures: int, output_path: str, samples: np.ndarray, simulation: np.ndarray, domain_paths: list[str]
) -> None:
    if num_mixtures == 1:
        # Single best predicted data mixture
        chosen_mixtures = [samples[np.argmax(simulation)]]
    else:
        # Sample diverse mixtures from the top-k pool by predicted score
        k_pool = max(num_mixtures * num_mixtures, 128)
        top_pool = samples[np.argsort(simulation)[-k_pool:]]
        chosen_mixtures = top_pool[np.random.choice(len(top_pool), size=num_mixtures, replace=False)]  # noqa: NPY002

    # Save n1.sh, ..., n{num_mixtures}.sh files
    for i, mixture in enumerate(chosen_mixtures):
        with open(os.path.join(output_path, f"n{i + 1}.sh"), "w") as f:
            f.write("#!/bin/bash\n")
            f.write("cat <<EOF\n")
            for path, weight in zip(domain_paths, mixture, strict=True):
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

    if (args.input_paths is None) != (args.mixtures_paths is None):
        msg = "--input-paths and --mixtures-paths must be provided together"
        raise ValueError(msg)
    if args.input_paths is not None and args.lm_harness_results_csv_path is not None:
        msg = "Pass either --input-paths/--mixtures-paths or --lm-harness-results-csv-path, not both"
        raise ValueError(msg)
    if args.num_samples < args.num_mixtures:
        msg = (
            f"--num-samples ({args.num_samples}) must be >= --num-mixtures ({args.num_mixtures}); "
            f"cannot draw more diverse mixtures than the candidate pool."
        )
        raise ValueError(msg)

    if args.input_paths is not None:
        df = load_benchmark_results(
            input_paths=args.input_paths,
            domain_names=domain_names,
            mixtures_paths=args.mixtures_paths,
            output_path=args.output_path,
        )
    elif args.lm_harness_results_csv_path is not None:
        df = pd.read_csv(args.lm_harness_results_csv_path)
    else:
        msg = "Either (--input-paths, --mixtures-paths) or (--lm-harness-results-csv-path) must be provided"
        raise ValueError(msg)

    if args.metric not in df.columns:
        msg = f"--metric {args.metric!r} is not a column in the benchmark results. Available: {sorted(df.columns)}"
        raise ValueError(msg)

    # Force numeric dtype on feature/target columns; the per-row Series construction in
    # load_benchmark_results leaves them as object, which modern LightGBM rejects.
    numeric_cols = [*domain_names, "arc_easy_acc", "piqa_acc_norm", "hellaswag_acc_norm", "valid_avg"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    predictor = fit_predictor(df, domain_names, args.metric)

    # Get the token distribution of each domain
    token_dist = get_token_distribution(args.domains_path)
    # Floor the alphas so a zero-byte .bin doesn't break np.random.dirichlet (requires alpha > 0)
    prior_dist = np.maximum(np.array([token_dist[str(f)] for f in bin_files]), 1e-12)
    domain_paths = [str(f.with_suffix("")) for f in bin_files]

    samples = np.random.dirichlet(prior_dist, args.num_samples)  # noqa: NPY002
    simulation = predictor.predict(samples)

    generate_mixtures(args.num_mixtures, args.output_path, samples, simulation, domain_paths)


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # I/O args
    parser.add_argument("--input-paths", type=str, nargs="+", required=False)
    parser.add_argument("--lm-harness-results-csv-path", type=str, required=False)
    parser.add_argument("--domains-path", type=str, required=True)
    parser.add_argument("--mixtures-paths", type=str, nargs="+", required=False)
    parser.add_argument("--output-path", type=str, required=True)

    # Prediction args
    parser.add_argument("--metric", type=str, default="valid_avg")
    parser.add_argument("--num-mixtures", type=int, default=1)
    parser.add_argument(
        "--num-samples", type=int, default=100000, help="Dirichlet samples to score for mixture search"
    )

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
