"""Typed configuration loading and validation."""

from .loader import ConfigError, GalaConfig, load_config

__all__ = ["ConfigError", "GalaConfig", "load_config"]
