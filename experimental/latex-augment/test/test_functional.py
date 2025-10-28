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

from latex_augment import functional as F
from latex_augment.latex_parser import LatexDocument


def test_remove_bibtex():
    # Test removing \bibliography{...}
    doc = LatexDocument(
        rb"""
\documentclass{article}
\begin{document}
Some text here.
\bibliography{references}
More text here.
\end{document}
"""
    )
    doc = F.remove_bibliography(doc)
    assert b"\\bibliography{references}" not in doc.source
    assert b"Some text here." in doc.source
    assert b"More text here." in doc.source


def test_remove_bib():
    # Test removing \begin{thebibliography}...\end{thebibliography}
    doc = LatexDocument(
        rb"""
\documentclass{article}
\begin{document}
Some text here.
\begin{thebibliography}{9}
\bibitem{key1} Author, Title
\bibitem{key2} Author2, Title2
\end{thebibliography}
More text here.
\end{document}
"""
    )
    doc = F.remove_bibliography(doc)
    assert b"\\begin{thebibliography}" not in doc.source
    assert b"\\bibitem{key1}" not in doc.source
    assert b"\\end{thebibliography}" not in doc.source
    assert b"Some text here." in doc.source
    assert b"More text here." in doc.source

    # Test document with no bibliography
    doc = LatexDocument(
        rb"""
\documentclass{article}
\begin{document}
Just some text here.
\end{document}
"""
    )
    doc_without_bib = F.remove_bibliography(doc)
    assert doc_without_bib.source == doc.source
