# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import random

import fasttext
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from nemo_curator.utils.file_utils import get_all_files_paths_under


def main(args):

    # Set the random seed for shuffling
    random.seed(args.seed)

    # Read in all samples into a list and shuffle the samples
    documents = []
    for ifile in get_all_files_paths_under(args.fasttext_files_dir):
        if os.path.splitext(ifile)[-1] == ".txt":
            with open(ifile, "r") as fp:
                documents += fp.readlines()
    random.shuffle(documents)

    # Get total number of samples
    nlines = len(documents)

    num_train = round(nlines * args.validation_split)

    # Split into training and validation samples
    train_samples = documents[:num_train]
    valid_samples = documents[num_train:]

    # Write out the training and validation samples
    with open(args.output_train_file, "w") as fp:
        for document in train_samples:
            fp.write(document)

    with open(args.output_validation_file, "w") as fp:
        for document in valid_samples:
            fp.write(document)

    # Train the model
    model = fasttext.train_supervised(
        input=args.output_train_file,
        lr=args.learning_rate,
        dim=args.word_vector_dim,
        epoch=args.num_epochs,
        wordNgrams=args.wordNgrams,
    )

    # Save the classifier as a FastText model
    model.save_model(args.output_model)

    if args.output_predictions is not None:
        fout = open(args.output_predictions, "wb")

    # Read in the model and compute accuracy and other metrics on the data
    hq_label = args.high_quality_label
    prds, lbls = [], []
    with open(args.output_validation_file, "r") as f:
        for line in tqdm(f.readlines()):
            # Split the text and the label
            label_t, doc = line.split(" ", 1)
            doc = doc.rstrip()
            labels_p, scores = model.predict(doc, k=2)
            # Write the predictions to file
            if args.output_predictions is not None:
                line = {
                    "text": doc,
                    "label": label_t,
                    f"{labels_p[0]}": scores[0],
                    f"{labels_p[1]}": scores[1],
                }
                myjson = json.dumps(line, ensure_ascii=False)
                fout.write(myjson.encode("utf-8"))
                fout.write("\n".encode("utf-8"))
            # Save predictions and labels
            prds.append(1) if labels_p[0] == hq_label else prds.append(0)
            lbls.append(1) if label_t == hq_label else lbls.append(0)

    # Print out the metrics computed on the validation data
    tn, fp, fn, tp = confusion_matrix(prds, lbls).ravel()
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    print(f"Acc={accuracy} Prec={precision} Rec={recall} f1={f1}")


def attach_args(
    parser=argparse.ArgumentParser(
        """
Train a skip-gram quality classifier with FastText

Takes as input files with prepared samples for training
a skip-gram classifier with FastText, trains a skip-gram
classifier on the input samples and writes the trained classifier
out to disk as a FastText model.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    parser.add_argument(
        "--fasttext-files-dir",
        type=str,
        default=None,
        required=True,
        help="The input directory containing the file(s) "
        "containing the prepared FastText samples",
    )
    parser.add_argument(
        "--high-quality-label",
        type=str,
        default="__label__hq",
        help="The label assigned to the high quality samples "
        "when preparing the data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1992,
        help="The seed used for randomly shuffling the documents",
    )
    parser.add_argument(
        "--output-train-file",
        type=str,
        default="./fasttext_samples.train",
        help="The concatenated, shuffled samples used "
        "to train the skip-gram classifier",
    )
    parser.add_argument(
        "--output-validation-file",
        type=str,
        default="./fasttext_samples.valid",
        help="The concatenated, shuffled samples used to "
        "for computing validation metrics",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.9,
        help="The training validation split",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=None,
        required=True,
        help="The output trained skip-gram classifier written " "as a FastText model",
    )
    parser.add_argument(
        "--wordNgrams",
        type=int,
        default=2,
        help="The size of the word n-gram used to train the classifier "
        "(default is bigram)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="The learning rate used to train the classifier",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of epochs used to train the classifier",
    )
    parser.add_argument(
        "--word-vector-dim",
        type=int,
        default=100,
        help="Size of word vectors to be computed by the model",
    )
    parser.add_argument(
        "--output-predictions",
        type=str,
        default=None,
        help="The output predictions on the validation data. "
        "If a file is not specified, the predictions are not "
        "written to file",
    )
    return parser


def console_script():
    main(attach_args().parse_args())
