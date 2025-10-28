# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import click
import torch
from contextlib import contextmanager
from . import transforms as T
from .latex_parser import LatexDocument
from .latex_translator import LatexError, prepare_document, translate_split_document
from .mbart_translator import MBartTranslatorFromEnglish
from .triton_translator import TritonTranslator


@click.group()
def main():
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("outpath", type=click.Path(), required=False)
def augment(path, outpath):
    if outpath is None:
        outpath = path.replace(".tex", "_augmented.tex")

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )

    augment = T.Compose(
        [
            T.RandomPageSizeAndMargins(p=0.5),
            T.RandomLineSpacing(p=0.3),
            T.RandomWordSpacing(p=0.3),
            T.RandomLetterSpacing(p=0.3),
            T.RandomFontSize(p=0.2),
            T.RandomTextAlignment(p=0.3),
            T.RandomColumnLayout(p=0.5),
            T.RandomSubsectionColumnLayout(p=0.3),
            T.RandomFloatRotation(p=0.2),
            T.RandomTableColumnSeparators(p=0.5),
            T.RemoveBibliography(p=0.5),
            T.RandomFont(p=0.9),
            T.RandomSubsectionTextColor(p=0.2),
            T.RandomSepiaPageColor(p=0.2),
            T.RandomPageColor(p=0.1),
            T.RandomInvertedColors(p=0.3),
            T.RandomPageBackground(p=0.5),
            T.RandomTextColor(p=0.3),
        ]
    )

    doc = LatexDocument.from_file(path)
    doc, pdf = augment.check(doc, dirname=os.path.dirname(path))

    with open(outpath, "wb") as f:
        f.write(doc.source)
    with open(outpath.replace(".tex", ".pdf"), "wb") as f:
        f.write(pdf)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("outpath", type=click.Path(), required=False)
@click.option("--verbose", "-v", is_flag=True)
def prepare(path, outpath, verbose):
    """Split LaTeX document into sentences for translation."""
    if outpath is None:
        outpath = path.replace(".tex", f"_parse.tex")

    logging.basicConfig(
        format="%(message)s",
        filename=outpath.replace(".tex", ".log"),
        filemode="w",
        level=logging.INFO,
    )
    if verbose:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.getLogger().addHandler(h)
        logging.getLogger().setLevel(logging.INFO)

    try:
        doc = LatexDocument.from_file(path)
        parsed = prepare_document(doc, document=True, verbose=verbose)
    except LatexError as e:
        print(f"{path}: {e}", file=sys.stderr)
        sys.exit(1)
    with open(outpath, "w") as f:
        f.write(parsed)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("outpath", type=click.Path(), required=False)
@click.option("--lang", default="zh")
@click.option("--triton-server", type=str, default=None)
@click.option("--verbose", "-v", is_flag=True)
def translate(
    path,
    outpath,
    lang,
    triton_server,
    verbose,
):
    """Translate LaTeX document from parsed document."""
    if outpath is None:
        outpath = path.replace(".tex", f"_{lang}.tex")

    logging.basicConfig(
        format="%(message)s",
        filename=outpath.replace(".tex", "_translate.log"),
        filemode="w",
        level=logging.INFO,
    )
    if verbose:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.getLogger().addHandler(h)
        logging.getLogger().setLevel(logging.INFO)

    # mitigate repeated hallucinations with repetition penalty, e.g.
    # 32、16、16、8、8、8、8、8、8、8、8、12、12、12、12、12、12、12、12、12、...
    repetition_penalty = 1.1

    if lang == "none":

        @contextmanager
        def translator_manager(*args, **kwargs):
            yield lambda batch: batch
    elif triton_server:

        @contextmanager
        def translator_manager(*args, **kwargs):
            yield TritonTranslator(*args, **kwargs, server_address=triton_server)
    else:

        @contextmanager
        def translator_manager(*args, **kwargs):
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            yield MBartTranslatorFromEnglish(*args, **kwargs, device=device)

    with open(path, "r") as f:
        latex = f.read()
    with translator_manager(lang, repetition_penalty=repetition_penalty) as translator:
        try:
            with open(outpath + ".tmp", "wb", buffering=0) as f:
                for chunk in translate_split_document(latex, lang=lang, translator=translator):
                    f.write(chunk.encode("utf8"))
        except LatexError as e:
            os.remove(outpath + ".tmp")
            print(f"{path}: {e}", file=sys.stderr)
            sys.exit(1)
        else:
            os.rename(outpath + ".tmp", outpath)


if __name__ == "__main__":
    main()
