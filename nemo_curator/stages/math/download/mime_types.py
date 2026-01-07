# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# MIME types from HTTP headers that indicate text content (no HTML extraction needed)
TEXT_MIME_TYPES: set[str] = {
    "text/x-web-markdown",
    "text/x-verilog",
    "text/x-rst",
    "text/x-ruby",
    "text/x-rsrc",
    "text/x-python",
    "text/x-perl",
    "text/x-pascal",
    "text/x-objcsrc",
    "text/x-ml",
    "text/x-matlab",
    "text/x-log",
    "text/x-haskell",
    "text/x-fortran",
    "text/x-expect",
    "text/x-diff",
    "text/x-csrc",
    "text/x-common-lisp",
    "text/x-chdr",
    "text/x-cgi",
    "text/x-c++src",
    "text/x-basic",
    "text/vtt",
    "text/x-assembly",
    "text/troff",
    "text/plain",
    "message/rfc822",
    "message/news",
    "application/mathematica",
    "application/mbox",
    "application/postscript",
    "application/x-elc",
    "application/x-matlab-data",
    "application/x-sas",
    "application/x-sh",
    "application/x-subrip",
    "application/x-tex",
    "application/x-tika-msoffice",
}

# MIME types from HTTP headers that indicate HTML content (needs extraction)
HTML_MIME_TYPES: set[str] = {
    "text/x-php",
    "text/x-jsp",
    "text/x-coldfusion",
    "text/html",
    "message/x-emlx",
    "text/asp",
    "image/svg+xml",
    "application/xml",
    "application/atom+xml",
    "application/rdf+xml",
    "application/rss+xml",
    "application/x-bibtex-text-file",
    "application/xhtml+xml",
}

# Magic MIME types (from libmagic) that indicate text content
TEXT_MAGIC_TYPES: set[str] = {
    "text/x-shellscript",
    "text/x-perl",
    "text/x-lisp",
    "text/x-java",
    "text/x-fortran",
    "text/x-diff",
    "application/postscript",
    "application/x-matlab-data",
    "message/news",
    "message/rfc822",
    "text/plain",
    "text/texmacs",
    "text/x-Algol68",
}

# Magic MIME types (from libmagic) that indicate HTML content
HTML_MAGIC_TYPES: set[str] = {
    "text/xml",
    "text/x-tex",
    "text/x-php",
    "text/x-ruby",
    "text/x-script.python",
    "text/x-objective-c",
    "text/x-forth",
    "text/x-c",
    "text/x-c++",
    "text/csv",
    "text/html",
    "application/octet-stream",
    "application/x-appleworks3",
    "application/x-bytecode.python",
    "application/x-setupscript",
    "application/x-wine-extension-ini",
    "image/svg+xml",
}
