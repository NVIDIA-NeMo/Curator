from nemo_curator.stages.text.utils.constants import regex_url


def test_regex_url_does_not_treat_caret_as_valid_url_character() -> None:
    text = "Visit http://exa^mple.com for details"

    assert regex_url.findall(text) == ["http://exa"]
