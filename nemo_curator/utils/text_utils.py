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

import ast
import os
import string
import tokenize
import warnings
from io import StringIO
from itertools import groupby


def get_word_splitter(language):
    language = language.lower()
    if language == "zh":
        import jieba

        return jieba.cut
    else:
        return default_splitter


def default_splitter(document):
    return document.split()


def get_paragraphs(document):
    # Split the document into paragraphs.
    # A paragraph is defined as a sequence of lines
    # separated by a double newline.
    return document.split("\n\n")


def get_sentences(document):
    # Split the document into sentences.
    # A sentence is defined as a sequence of lines separated
    # by a single newline.
    return [x for x in document.split("\n") if len(x.strip()) > 0]


def get_ngrams(input_list, n):
    # Fast function to return n-grams from a list of tokens.
    return [item for item in zip(*[input_list[i:] for i in range(n)])]


def is_paragraph_indices_in_top_or_bottom_only(
    boilerplate_paragraph_indices,
    num_paragraphs,
):

    def _is_contiguous(indices):
        # Indices are sorted in ascending order.
        num_indices = len(indices) - 1
        return all(indices[i] + 1 == indices[i + 1] for i in range(num_indices))

    # See if the indices are contiguous and exclusively at the top/bottom.
    # Indices are sorted in ascending order.
    # If num_paragraphs = 11:
    # Valid indices example : [0, 1, 9, 10]
    # Invalid indices example : [0, 1, 3, 9, 10]
    # Invalid indices example : [0, 1, 3, 5, 6, 9, 10]
    # Invalid indices example : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if len(boilerplate_paragraph_indices) == num_paragraphs:
        return False
    return _is_contiguous(boilerplate_paragraph_indices) and (
        boilerplate_paragraph_indices[0] == 0
        or boilerplate_paragraph_indices[-1] == num_paragraphs - 1
    )


# Node types for processing abstract syntax tree
NODE_TYPES = {
    ast.ClassDef: "Class",
    ast.FunctionDef: "Function/Method",
    ast.Module: "Module",
}


def get_comments_and_docstring(source, comments=True, clean_comments=False):
    """
    Extract all natural text in source: comments + doctsrings
      the extraction fails in case of syntax errors in the file
      Args:
          source: the code to parse
          comments: if True extract comments two
          clean_comment: if True remove # from extracted comments
      Returns:
          a string with concatenated docstrings and comments
    """

    try:
        docstrings = "\n".join(get_docstrings(source))
    except Exception:
        docstrings = None
        warnings.warn(
            "code couldn't be parsed due to compilation failure, "
            "no docstring is extracted"
        )

    if comments:
        try:
            comments = get_comments(source, clean=clean_comments)
        except Exception:
            comments = None
            warnings.warn("tokenization error, no comment is extracted")
    else:
        comments = ""

    return docstrings, comments


def get_comments(s, clean=False):
    "Returns a string including all coments"
    coments = []
    g = tokenize.generate_tokens(StringIO(s).readline)
    for toknum, tokval, _, _, _ in g:
        # print(toknum,tokval)
        if toknum == tokenize.COMMENT:
            coments.append((toknum, tokval))
    result = tokenize.untokenize(coments)
    if clean:
        result = result.replace("#", "")
    return result


def get_docstrings(source, module="<string>"):
    """Parse Python source code from file or string and print docstrings."""
    if hasattr(source, "read"):
        filename = getattr(source, "name", module)
        module = os.path.splitext(os.path.basename(filename))[0]
        source = source.read()

    docstrings = sorted(
        parse_docstrings(source), key=lambda x: (NODE_TYPES.get(type(x[0])), x[1])
    )

    grouped = groupby(docstrings, key=lambda x: NODE_TYPES.get(type(x[0])))
    results = []
    for _, group in grouped:
        for _, name, docstring in group:
            name = name if name else module
            if docstring:
                results.append(docstring)
    return results


def parse_docstrings(source):
    """Parse Python source code and yield a tuple of ast node instance, name,
    and docstring for each function/method, class and module."""
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, tuple(NODE_TYPES)):
            docstring = ast.get_docstring(node)

            yield (node, getattr(node, "name", None), docstring)


def remove_punctuation(str_in):
    return str_in.translate(str_in.maketrans("", "", string.punctuation))


def get_words(text):
    word_start_char_positions = []
    prev = 0
    words = []

    text = text.lower()
    text = remove_punctuation(text)
    if len(text) > 0:
        for i in range(len(text)):
            if text[i] != " ":
                if i == 0 or text[i - 1] == " ":
                    word_start_char_positions.append(i)
                    if i != 0:
                        words.append(text[prev:i].strip())
                    prev = i
        words.append(text[prev : i + 1].strip())
        if words[0] == "":
            words = words[1:]
    return words, word_start_char_positions
