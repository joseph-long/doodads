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

config = partial(dataclasses.dataclass, kw_only=True)

def convert_bool(x):
    if x in (True, False):
        return x
    if x.lower() in ('true', 't', '1'):
        return True
    elif x.lower() in ('false', 'f', '0'):
        return False
    else:
        raise ValueError(f"Cannot convert {repr(x)} to boolean")


def convert_list(x, convert_type=lambda x: x):
    if isinstance(x, str):
        # handle escaped commas by temporarily replacing with an escaped value
        x = x.replace(r'\,', '\u0001')
        out = []
        for y in x.split(','):
            val = y.replace('\u0001', ',')
            try:
                val = convert_type(val)
                out.append(val)
            except ValueError:
                raise ValueError(f"Could not apply type conversion to {repr(val)} from list {repr(x)}")
    else:
        # type conversion was already handled and x is a list at this point
        assert isinstance(x, list)
        out = x
    return out

TYPE_HOOKS = {
    bool: convert_bool,
    int: int,
    float: float,
    list[str]: partial(convert_list, convert_type=str),
    list[int]: partial(convert_list, convert_type=int),
    list[float]: partial(convert_list, convert_type=float),
    pathlib.Path: pathlib.Path,
}
from_dict = partial(dacite.from_dict, config=dacite.Config(strict=True, type_hooks=TYPE_HOOKS, cast=[Enum]))
from_dict.__doc__ = dacite.from_dict.__doc__

def merge_configs(config, *configs):
    def _merge_dict(original: Any, update: Any) -> dict:
        merged = {}
        for k in (original.keys() | update.keys()):
            if k in original and k in update:
                # merge
                original_subtree = original[k]
                update_subtree = update[k]
                if isinstance(original_subtree, Mapping) and isinstance(update_subtree, Mapping):
                    merged[k] = _merge_dict(original_subtree, update_subtree)
                else:
                    # use update if available, otherwise keep original
                    merged[k] = update.get(k, original.get(k))
                    assert merged[k] is not None, f"Should have gotten {k=} from at least one of the inputs"
            elif k in original:
                merged[k] = original[k]
            elif k in update:
                merged[k] = update[k]
            else:
                raise RuntimeError("Unreachable")
        return merged
    base = config
    for updates in configs:
        base =_merge_dict(base, updates)
    return base



def field(*args, **kwargs):
    metadata = None
    if "help" in kwargs:
        metadata = {"help": kwargs.pop("help")}
    return dataclasses.field(*args, metadata=metadata, **kwargs)
