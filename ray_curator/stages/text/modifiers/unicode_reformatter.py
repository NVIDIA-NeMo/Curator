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

from typing import Literal

import ftfy
from ftfy import TextFixerConfig

from nemo_curator.stages.text.modifiers.doc_modifier import DocumentModifier


class UnicodeReformatter(DocumentModifier):
    def __init__(  # noqa: PLR0913
        self,
        config: TextFixerConfig | None = None,
        unescape_html: str | bool = "auto",
        remove_terminal_escapes: bool = True,
        fix_encoding: bool = True,
        restore_byte_a0: bool = True,
        replace_lossy_sequences: bool = True,
        decode_inconsistent_utf8: bool = True,
        fix_c1_controls: bool = True,
        fix_latin_ligatures: bool = False,
        fix_character_width: bool = False,
        uncurl_quotes: bool = False,
        fix_line_breaks: bool = False,
        fix_surrogates: bool = True,
        remove_control_chars: bool = True,
        normalization: Literal["NFC", "NFD", "NFKC", "NFKD"] | None = None,
        max_decode_length: int = 1000000,
        explain: bool = True,
    ):
        """
        See https://ftfy.readthedocs.io/en/latest/config.html for more information about the TextFixerConfig parameters.

        Args:
            config: A TextFixerConfig object. If None, the default values for all other arguments will be used.
                Default is None.
            unescape_html (str, bool): Default is None.
                Configures whether to replace HTML entities such as &amp; with the character
                they represent. "auto" says to do this by default, but disable it when a
                literal < character appears, indicating that the input is actual HTML and
                entities should be preserved. The value can be True, to always enable this
                fixer, or False, to always disable it.
            remove_terminal_escapes (bool): Default is True.
                Removes "ANSI" terminal escapes, such as for changing the color of text in a
                terminal window.
            fix_encoding (bool): Default is True.
                Detect mojibake and attempt to fix it by decoding the text in a different
                encoding standard.

            The following four options affect `fix_encoding` works, and do nothing if
            `fix_encoding` is False:
                restore_byte_a0 (bool): Default is True.
                    Allow a literal space (U+20) to be interpreted as a non-breaking space
                    (U+A0) when that would make it part of a fixable mojibake string.
                    Because spaces are very common characters, this could lead to false
                    positives, but we try to apply it only when there's strong evidence for
                    mojibake. Disabling `restore_byte_a0` is safer from false positives,
                    but creates false negatives.
                replace_lossy_sequences (bool): Default is True.
                    Detect mojibake that has been partially replaced by the characters
                    '�' or '?'. If the mojibake could be decoded otherwise, replace the
                    detected sequence with '�'.
                decode_inconsistent_utf8 (bool): Default is True.
                    When we see sequences that distinctly look like UTF-8 mojibake, but
                    there's no consistent way to reinterpret the string in a new encoding,
                    replace the mojibake with the appropriate UTF-8 characters anyway.
                    This helps to decode strings that are concatenated from different
                    encodings.
                fix_c1_controls (bool): Default is True.
                    Replace C1 control characters (the useless characters U+80 - U+9B that
                    come from Latin-1) with their Windows-1252 equivalents, like HTML5 does,
                    even if the whole string doesn't decode as Latin-1.

            fix_latin_ligatures (bool): Default is False.
                Replace common Latin-alphabet ligatures, such as ``ﬁ``, with the
                letters they're made of.
            fix_character_width (bool): Default is False.
                Replace fullwidth Latin characters and halfwidth Katakana with
                their more standard widths.
            uncurl_quotes (bool): Default is False.
                Replace curly quotes with straight quotes.
            fix_line_breaks (bool): Default is False.
                Replace various forms of line breaks with the standard Unix line
                break, ``\n``.
            fix_surrogates (bool): Default is True.
                Replace sequences of UTF-16 surrogate codepoints with the character
                they were meant to encode. This fixes text that was decoded with the
                obsolete UCS-2 standard, and allows it to support high-numbered
                codepoints such as emoji.
            remove_control_chars (bool): Default is True.
                Remove certain control characters that have no displayed effect on text.
            normalization ("NFC", "NFD", "NFKC", "NFKD", None): Default is None.
                Choose what kind of Unicode normalization is applied. Usually, we apply
                NFC normalization, so that letters followed by combining characters become
                single combined characters.
                Changing this to "NFKC" applies more compatibility conversions, such as
                replacing the 'micro sign' with a standard Greek lowercase mu, which looks
                identical. However, some NFKC normalizations change the meaning of text,
                such as converting "10³" to "103".
                `normalization` can be None, to apply no normalization.
            max_decode_length (int): Default is 1000000.
                The maximum size of "segment" that ftfy will try to fix all at once.
            explain (bool): Default is True.
                Whether to compute 'explanations', lists describing what ftfy changed.
                When this is False, the explanation will be None, and the code that
                builds the explanation will be skipped, possibly saving time.
                Functions that accept TextFixerConfig and don't return an explanation
                will automatically set `explain` to False.

        """
        if config is None:
            self.config = TextFixerConfig(
                unescape_html=unescape_html,
                remove_terminal_escapes=remove_terminal_escapes,
                fix_encoding=fix_encoding,
                restore_byte_a0=restore_byte_a0,
                replace_lossy_sequences=replace_lossy_sequences,
                decode_inconsistent_utf8=decode_inconsistent_utf8,
                fix_c1_controls=fix_c1_controls,
                fix_latin_ligatures=fix_latin_ligatures,
                fix_character_width=fix_character_width,
                uncurl_quotes=uncurl_quotes,
                fix_line_breaks=fix_line_breaks,
                fix_surrogates=fix_surrogates,
                remove_control_chars=remove_control_chars,
                normalization=normalization,
                max_decode_length=max_decode_length,
                explain=explain,
            )
        else:
            self.config = config

        super().__init__()

    def modify_document(self, text: str) -> str:
        return ftfy.fix_text(text, config=self.config)
