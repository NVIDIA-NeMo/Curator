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

import string

from .latex_characters import get_unicode_charset

# don't add hyphens so that it is only used for line wrapping and
# can be ignored in evaluation
ASCII_PUNCTUATION = string.punctuation.replace("-", "")
CHINESE_PUNCTUATION = (
    ASCII_PUNCTUATION + 3 * '。，、…—·？！；：「」『』"（）【】〖〗〔〕《》〈〉'
)
ASCII_NUMBERS = string.digits
# numerical symbols from GB2312
# CHINESE_NUMBERS = (
#     "⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛"
#     "⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇①②③④⑤⑥⑦⑧⑨⑩"
#     "㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩"
#     "ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ"
# )

LATEX_SYMBOLS = """
\\textasciicircum
\\textasciitilde
\\textasteriskcentered
\\textbackslash
\\textbar
\\textbraceleft∗
\\textbraceright∗
\\textbullet
\\textcopyright∗
\\textdagger∗
\\textdaggerdbl∗
\\textdollar∗
\\textellipsis∗
\\textemdash
\\textendash
\\textexclamdown
\\textgreater
\\textless
\\textordfeminine
\\textordmasculine
\\textparagraph∗
\\textperiodcentered
\\textquestiondown
\\textquotedblleft
\\textquotedblright
\\textquoteleft
\\textquoteright
\\textregistered
\\textsection∗
\\textsterling∗
\\texttrademark
\\textunderscore∗
\\textvisiblespace
""".split()


MULTISCRIPT_BLOCKS = {
    "greek": [
        "Greek and Coptic",
        "Greek Extended",
    ],
    
    "cyrillic": [
        "Cyrillic",
        "Cyrillic Supplement",
        "Cyrillic Extended-A",
        "Cyrillic Extended-B",
        "Cyrillic Extended-C",
        "Cyrillic Extended-D",
    ],
    
    "armenian": [
        "Armenian",
    ],
    
    "georgian": [
        "Georgian",
        "Georgian Extended",
        "Georgian Supplement",
    ],
    
    "european_ancient": [
        "Glagolitic",
        "Glagolitic Supplement",
        "Gothic",
        "Old Permic",
        "Old Hungarian",
        "Runic",
        "Ogham",
    ],
    
    "hebrew": [
        "Hebrew",
    ],
    
    "arabic": [
        "Arabic",
        "Arabic Supplement",
        "Arabic Extended-A",
        "Arabic Extended-B",
        "Arabic Extended-C",
        "Arabic Presentation Forms-A",
        "Arabic Presentation Forms-B",
        "Arabic Mathematical Alphabetic Symbols",
    ],
    
    "middle_eastern": [
        "Syriac",
        "Syriac Supplement",
        "Thaana",
        "NKo",
        "Samaritan",
        "Mandaic",
        "Imperial Aramaic",
        "Palmyrene",
        "Nabataean",
        "Hatran",
        "Phoenician",
        "Lydian",
        "Old Persian",
        "Ugaritic",
        "Manichaean",
        "Avestan",
        "Inscriptional Parthian",
        "Inscriptional Pahlavi",
        "Psalter Pahlavi",
        "Old Turkic",
    ],
    
    "devanagari": [
        "Devanagari",
        "Devanagari Extended",
        "Devanagari Extended-A",
    ],
    
    "bengali": [
        "Bengali",
    ],
    
    "tamil": [
        "Tamil",
        "Tamil Supplement",
    ],
    
    "sinhala": [
        "Sinhala",
        "Sinhala Archaic Numbers",
    ],
    
    "south_asian": [
        "Gurmukhi",
        "Gujarati",
        "Oriya",
        "Telugu",
        "Kannada",
        "Malayalam",
        "Brahmi",
        "Kaithi",
        "Mahajani",
        "Sharada",
        "Khojki",
        "Multani",
        "Khudawadi",
        "Grantha",
        "Newa",
        "Tirhuta",
        "Siddham",
        "Modi",
        "Takri",
        "Dogra",
        "Nandinagari",
        "Bhaiksuki",
        "Marchen",
        "Masaram Gondi",
        "Gunjala Gondi",
    ],
    
    "thai": [
        "Thai",
    ],
    
    "lao": [
        "Lao",
    ],
    
    "tibetan": [
        "Tibetan",
    ],
    
    "myanmar": [
        "Myanmar",
        "Myanmar Extended-A",
        "Myanmar Extended-B",
        "Myanmar Extended-C",
    ],
    
    "khmer": [
        "Khmer",
        "Khmer Symbols",
    ],
    
    "mongolian": [
        "Mongolian",
        "Mongolian Supplement",
    ],
    
    "southeast_asian": [
        "Limbu",
        "Tai Le",
        "New Tai Lue",
        "Buginese",
        "Tai Tham",
        "Balinese",
        "Sundanese",
        "Sundanese Supplement",
        "Batak",
        "Lepcha",
        "Ol Chiki",
        "Sora Sompeng",
        "Chakma",
        "Warang Citi",
        "Pau Cin Hau",
        "Mro",
        "Ahom",
        "Tai Viet",
        "Meetei Mayek",
        "Meetei Mayek Extensions",
    ],
    
    "cjk": [
        "CJK Radicals Supplement",
        "Kangxi Radicals",
        "Ideographic Description Characters",
        "CJK Symbols and Punctuation",
        "CJK Unified Ideographs Extension A",
        "CJK Unified Ideographs",
        "CJK Compatibility",
        "CJK Unified Ideographs Extension B",
        "CJK Unified Ideographs Extension C",
        "CJK Unified Ideographs Extension D",
        "CJK Unified Ideographs Extension E",
        "CJK Unified Ideographs Extension F",
        "CJK Unified Ideographs Extension G",
        "CJK Unified Ideographs Extension H",
        "CJK Unified Ideographs Extension I",
        "CJK Compatibility Ideographs",
        "CJK Compatibility Ideographs Supplement",
        "CJK Compatibility Forms",
        "Enclosed CJK Letters and Months",
    ],
    
    "hiragana": [
        "Hiragana",
    ],
    
    "katakana": [
        "Katakana",
        "Katakana Phonetic Extensions",
        "Kana Supplement",
        "Kana Extended-A",
        "Small Kana Extension",
    ],
    
    "hangul": [
        "Hangul Jamo",
        "Hangul Jamo Extended-A",
        "Hangul Jamo Extended-B",
        "Hangul Compatibility Jamo",
        "Hangul Syllables",
    ],
    
    "bopomofo": [
        "Bopomofo",
        "Bopomofo Extended",
        "Kanbun",
    ],
    
    "yi": [
        "Yi Syllables",
        "Yi Radicals",
    ],
    
    "nushu": [
        "Nushu",
    ],
    
    "ethiopic": [
        "Ethiopic",
        "Ethiopic Supplement",
        "Ethiopic Extended",
        "Ethiopic Extended-A",
        "Ethiopic Extended-B",
    ],
    
    "african": [
        "Tifinagh",
        "Bamum",
        "Bamum Supplement",
        "Mende Kikakui",
        "Adlam",
        "Bassa Vah",
        "Vai",
        "Osmanya",
        "Coptic",
        "Coptic Epact Numbers",
        "Meroitic Hieroglyphs",
        "Meroitic Cursive",
        "Old South Arabian",
        "Old North Arabian",
    ],
    
    "cherokee": [
        "Cherokee",
        "Cherokee Supplement",
    ],
    
    "canadian_aboriginal": [
        "Unified Canadian Aboriginal Syllabics",
        "Unified Canadian Aboriginal Syllabics Extended",
        "Unified Canadian Aboriginal Syllabics Extended-A",
    ],
    
    "american": [
        "Deseret",
        "Shavian",
        "Osage",
    ],
    
    "philippine": [
        "Tagalog",
        "Hanunoo",
        "Buhid",
        "Tagbanwa",
    ],
    
    "ancient_scripts": [
        "Linear B Syllabary",
        "Linear B Ideograms",
        "Linear A",
        "Phaistos Disc",
        "Lycian",
        "Carian",
        "Old Italic",
        "Cypriot Syllabary",
        "Kharoshthi",
        "Cuneiform",
        "Cuneiform Numbers and Punctuation",
        "Early Dynastic Cuneiform",
        "Egyptian Hieroglyphs",
        "Egyptian Hieroglyph Format Controls",
        "Egyptian Hieroglyphs Extended-A",
        "Anatolian Hieroglyphs",
    ],
    
    "specialized_scripts": [
        "Duployan",
        "Pahawh Hmong",
        "Nyiakeng Puachue Hmong",
        "Miao",
        "Tangut",
        "Tangut Components",
        "Tangut Supplement",
        "Wancho",
        "Zanabazar Square",
        "Soyombo",
        "Makasar",
        "Elbasan",
        "Caucasian Albanian",
        "Vithkuqi",
        "Todhri",
        "Toto",
        "Nag Mundari",
        "Ol Onal",
        "Sunuwar",
        "Kirat Rai",
        "Tangsa",
        "Gurung Khema",
        "Kawi",
        "Lisu",
        "Lisu Supplement",
        "Hanifi Rohingya",
        "Yezidi",
        "Chorasmian",
        "Dives Akuru",
        "Khitan Small Script",
        "Cypro-Minoan",
        "Old Sogdian",
        "Sogdian",
        "Old Uyghur",
        "Elymaic",
        "Garay",
        "Medefaidrin",
        "Tulu-Tigalari",
    ],
}

SYMBOL_BLOCKS = [
    # Mathematical Symbols
    "Mathematical Operators",
    "Supplemental Mathematical Operators",
    "Miscellaneous Mathematical Symbols-A",
    "Miscellaneous Mathematical Symbols-B",
    "Mathematical Alphanumeric Symbols",
    "Arrows",
    "Supplemental Arrows-A",
    "Supplemental Arrows-B",
    "Supplemental Arrows-C",
    "Miscellaneous Symbols and Arrows",
    "Superscripts and Subscripts",
    "Number Forms",
    "Letterlike Symbols",
    
    # Currency Symbols
    "Currency Symbols",
    
    # Musical Symbols
    "Musical Symbols",
    "Byzantine Musical Symbols",
    "Ancient Greek Musical Notation",
    "Znamenny Musical Notation",
    
    # Technical Symbols
    "Miscellaneous Technical",
    "Control Pictures",
    "Optical Character Recognition",
    "Box Drawing",
    "Block Elements",
    "Geometric Shapes",
    "Geometric Shapes Extended",
    "Braille Patterns",
    "Sutton SignWriting",
    
    # Game Symbols
    "Mahjong Tiles",
    "Domino Tiles",
    "Playing Cards",
    "Chess Symbols",
    
    # Miscellaneous Symbols
    "General Punctuation",
    "Supplemental Punctuation",
    "Miscellaneous Symbols",
    "Dingbats",
    "Ornamental Dingbats",
    "Miscellaneous Symbols and Pictographs",
    "Transport and Map Symbols",
    "Alchemical Symbols",
    "Symbols for Legacy Computing",
    "Enclosed Alphanumerics",
    "Enclosed Alphanumeric Supplement",
    "Enclosed Ideographic Supplement",

    # Emojis
    "Emoticons",
    "Supplemental Symbols and Pictographs",
    "Symbols and Pictographs Extended-A",

    # Phonetic Symbols
    "IPA Extensions",
    "Phonetic Extensions",
    "Phonetic Extensions Supplement",
    # "Spacing Modifier Letters",
    # "Modifier Tone Letters",
    # "Combining Diacritical Marks",
    # "Combining Diacritical Marks Extended",
    # "Combining Diacritical Marks Supplement",
    # "Combining Diacritical Marks for Symbols",
    # "Combining Half Marks",
    "Vedic Extensions",
    
    # Ancient Symbols
    "Aegean Numbers",
    "Ancient Greek Numbers",
    "Ancient Symbols",
    
    # Specialized Symbols
    "Shorthand Format Controls",
    "Rumi Numeral Symbols",
    "Indic Siyaq Numbers",
    "Ottoman Siyaq Numbers",
    "Counting Rod Numerals",
    "Kaktovik Numerals",
    "Mayan Numerals",
    "Tai Xuan Jing Symbols",
    "Yijing Hexagram Symbols",
    "Small Form Variants",
    "Alphabetic Presentation Forms",
    # "Variation Selectors",
    "Vertical Forms",
    # "Tags",
    # "Variation Selectors Supplement",
    "Halfwidth and Fullwidth Forms",
    "Specials",
]

MULTISCRIPT_CHARSET = {
    script: get_unicode_charset(MULTISCRIPT_BLOCKS[script])
    for script in MULTISCRIPT_BLOCKS
}

SYMBOL_CHARSET = get_unicode_charset(SYMBOL_BLOCKS)
