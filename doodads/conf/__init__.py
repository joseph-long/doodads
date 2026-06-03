"""Load configuration from TOML files, command-line arguments
environment variables, and even environment variables pointed
at files whose contents get read (e.g. for secrets) and turn
them into dataclass instances
"""
from collections.abc import Mapping
import dataclasses
from functools import partial
import logging
import pathlib
from typing import Any
from enum import Enum

import dacite
from dacite import UnexpectedDataError, MissingValueError

__all__ = (
    'field',
    'config',
    'config_to_dict',
    'dict_to_toml',
    'config_to_toml',
    'ConfigMismatch',
    'Command',
)

log = logging.getLogger(__name__)
