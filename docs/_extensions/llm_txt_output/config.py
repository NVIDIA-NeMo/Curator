"""Configuration management for llm_txt_output extension."""

from typing import Any

from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.util import logging

logger = logging.getLogger(__name__)


def get_default_settings() -> dict[str, Any]:
    """Get default configuration settings for llm_txt_output extension."""
    return {
        "enabled": True,
        "exclude_patterns": ["_build", "_templates", "_static", "apidocs"],
        "verbose": True,
        "base_url": "",  # Base URL for absolute links (e.g., https://docs.example.com/latest)
        "max_content_length": 5000,  # Max chars in overview section (0 = no limit)
        "summary_sentences": 2,  # Number of sentences for summary
        "include_metadata": True,  # Include metadata section
        "include_headings": True,  # Include key sections from headings
        "include_related_links": True,  # Include internal links as related resources
        "max_related_links": 10,  # Max related links to include
        "card_handling": "simple",  # "simple" or "smart" for grid cards
        "clean_myst_artifacts": True,  # Remove MyST directive artifacts
        "generate_full_file": True,  # Generate llm-full.txt with all documentation
    }


def apply_config_defaults(settings: dict[str, Any]) -> dict[str, Any]:
    """Apply default values to settings dictionary."""
    defaults = get_default_settings()

    for key, default_value in defaults.items():
        if key not in settings:
            settings[key] = default_value

    return settings


def validate_config(_app: Sphinx, config: Config) -> None:
    """Validate configuration values."""
    settings = _ensure_settings_dict(config)
    settings = apply_config_defaults(settings)
    config.llm_txt_settings = settings

    _validate_core_settings(settings)
    _validate_content_limits(settings)
    _validate_boolean_settings(settings)


def _ensure_settings_dict(config: Config) -> dict[str, Any]:
    """Ensure settings is a valid dictionary."""
    settings = getattr(config, "llm_txt_settings", {})
    if not isinstance(settings, dict):
        logger.warning("llm_txt_settings must be a dictionary. Using defaults.")
        settings = {}
        config.llm_txt_settings = settings
    return settings


def _validate_core_settings(settings: dict[str, Any]) -> None:
    """Validate core configuration settings."""
    # Validate card handling mode
    valid_modes = ["simple", "smart"]
    mode = settings.get("card_handling", "simple")
    if mode not in valid_modes:
        logger.warning(f"Invalid card_handling '{mode}'. Using 'simple'. Valid options: {valid_modes}")
        settings["card_handling"] = "simple"

    # Validate exclude patterns
    patterns = settings.get("exclude_patterns", [])
    if not isinstance(patterns, list):
        logger.warning("exclude_patterns must be a list. Using default.")
        settings["exclude_patterns"] = ["_build", "_templates", "_static", "apidocs"]


def _validate_content_limits(settings: dict[str, Any]) -> None:
    """Validate content-related limit settings."""
    limit_settings = {
        "max_content_length": (5000, "5000 (0 = no limit)"),
        "summary_sentences": (2, "2"),
        "max_related_links": (10, "10"),
    }

    for setting, (default_val, description) in limit_settings.items():
        value = settings.get(setting, default_val)
        if not isinstance(value, int) or value < 0:
            logger.warning(f"Invalid {setting} '{value}'. Using {description}.")
            settings[setting] = default_val


def _validate_boolean_settings(settings: dict[str, Any]) -> None:
    """Validate boolean configuration settings."""
    bool_settings = [
        "enabled",
        "verbose",
        "include_metadata",
        "include_headings",
        "include_related_links",
        "clean_myst_artifacts",
    ]

    defaults = get_default_settings()
    for setting in bool_settings:
        if setting in settings and not isinstance(settings.get(setting), bool):
            logger.warning(f"Setting '{setting}' must be boolean. Using default.")
            settings[setting] = defaults[setting]
