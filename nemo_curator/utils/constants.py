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

import re

import regex

end_marks = (".", "?", "!", '"', "'")
ellipsis_marks = set(
    ["...", "[...]", "…", "(...)", "[…]", "-»", "read more..", "read more"]
)
policy_substrings = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "privacy overview",
    "use of cookies",
    "use cookies",
    "privacy & cookies policy",
    "privacy and cookies policy",
    "This website uses cookies to improve your experience while you "
    "navigate through the website. Out of these cookies, the cookies "
    "that are categorized as necessary are stored on your browser as they "
    "are essential for the working of basic functionalities of the website. "
    "We also use third-party cookies that help us analyze and understand how "
    "you use this website. These cookies will be stored in your browser only "
    "with your consent. You also have the option to opt-out of these "
    "cookies. But opting out of some of these cookies may have an effect "
    "on your browsing experience.".lower(),
    "Necessary cookies are absolutely essential for the website to "
    "function properly. This category only includes cookies that "
    "ensures basic functionalities and security features of the website. "
    "These cookies do not store any personal information.".lower(),
    "Any cookies that may not be particularly necessary for the website "
    "to function and is used specifically to collect user personal data "
    "via analytics, ads, other embedded contents are termed as non-necessary "
    "cookies. It is mandatory to procure user consent prior to running these "
    "cookies on your website.".lower(),
    "This site uses cookies, including for analytics, personalization, and "
    "advertising purposes. For more information or to change your "
    "cookie settings, click here.".lower(),
    "If you continue to browse this site without changing your cookie "
    "settings, you agree to this use. AcceptRead More".lower(),
]
white_space_list = ["\t", "\n", "\r", "\b", " "]
common_english_words = set(["the", "be", "to", "of", "and", "that", "have", "with"])
bullet_list = set(
    [
        "•",
        "‣",
        "⁃",
        "⁌",
        "⁍",
        "∙",
        "○",
        "●",
        "◘",
        "◦",
        "⦾",
        "⦿",
    ]
)

regex_alpha = regex.compile("[[:alpha:]]")
regex_digit = regex.compile("[[:digit:]]")
regex_alphanum = re.compile("[a-zA-Z0-9\n?!,.]")
regex_url = re.compile(
    "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|"
    "(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
regex_paren = re.compile(r"{|}|⟨|⟩|\[|\]|\(|\)")
regex_hash = re.compile("#+")
