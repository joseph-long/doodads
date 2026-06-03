import dataclasses
from enum import Enum
import pathlib
import re

import dacite
import dacite.types

def parse_environment_config(environ: dict[str, str]):
    '''Turn environment variable definitions of the form
    ``FOO_BAR_BAZ=1`` into ``{'foo': {'bar': {'baz': '1'}}}``.

    If a name ends with ``__FILE``, try to open and read said
    file. The value is then the contents of the file, decoded
    as UTF-8, with any leading or trailing whitespace stripped.

    (Coercion to specific types happens later when passing to dacite.)
    '''
    parsed_config = {}
    for env_var_name in environ:
        if env_var_name.endswith('__FILE'):
            file_path: pathlib.Path = pathlib.Path(environ[env_var_name])
            env_var_name = env_var_name.replace('__FILE', '')
            if not file_path.is_file():
                raise FileNotFoundError(f"Unable to access {file_path} to populate {env_var_name}")
            else:
                with file_path.open('r', encoding='utf8') as fh:
                    env_var_value = fh.read().strip()
        else:
            env_var_value = environ[env_var_name]
        config_path = [x.lower() for x in env_var_name.split('_')]
        base = parsed_config
        for part in config_path[:-1]:
            if part not in base:
                base[part] = {}
            base = base[part]
        base[config_path[-1]] = env_var_value
    return parsed_config



def test_parse_cli_overrides_0():
    test0 = {'foo': {'bar': {'baz': '1'}}}
    assert test0 == _apply_one_cli_override(
        {},
        'foo.bar.baz=1',
    )
def test_parse_cli_overrides_1():
    test1 = {
        'foo': ['1'],
    }
    assert test1 == _apply_one_cli_override(
        {'foo': []},
        "foo[0]=1",
    ), "Unable to append to empty list with indexing"
def test_parse_cli_overrides_2():
    test2 = {
        'foo': ['1', '1'],
    }
    assert test2 == parse_cli_overrides(
        {'foo': ['1']},
        ["foo[1]=1"],
    ), "Unable to append to non-empty list with indexing"
def test_parse_cli_overrides_3():
    test3 = {
        'foo': [['1', '2'], ['3', '4']],
    }
    assert test3 == parse_cli_overrides(
        {'foo': [['1'], ['3']]},
        ["foo[0][1]=2", "foo[1][1]=4"],
    ), "Unable to append in a nested list"
def test_parse_cli_overrides_4():
    test4 = {
        'foo': [{'thing': '2'}, {'thing': '2'}],
    }
    assert test4 == _apply_one_cli_override(
        {'foo': [{'thing': '1'}, {'thing': '2'}]},
        "foo[0].thing=2",
    ), "Unable to override a key within a list"

def _assign(base_config, name, indices: list[int], rhs):
    if not indices:
        base_config[name] = rhs
    else:
        base_config = base_config[name]
        *leading_indices, last_idx = indices
        while leading_indices:
            idx = leading_indices.pop(0)
            if idx == len(base_config):
                base_config.append([])
            elif idx > len(base_config):
                raise RuntimeError("Can't override because incorrect list item index")
            base_config = base_config[idx]
        if len(base_config) == last_idx:
            base_config.append(rhs)
        else:
            base_config[last_idx] = rhs

def _descend(base_config, key_index_tuples, rhs):
    next_key_index_tuple, *key_index_tuples = key_index_tuples
    name, indexing_expr = next_key_index_tuple

    if indexing_expr:
        if name not in base_config:
            base_config[name] = []
        matches = re.findall(r'\[(\d+)\]', indexing_expr)
        indices = [int(x) for x in matches]
    else:
        indices = []
    if not len(key_index_tuples):
        _assign(base_config, name, indices, rhs)
    else:
        if name not in base_config:
            base_config[name] = {}
        base_config = base_config[name]
        for idx in indices:
            if len(base_config) > idx:
                base_config = base_config[idx]
            elif len(base_config) == idx:
                base_config.append([])
            else:
                raise RuntimeError("Can't override because incorrect list item index")

        _descend(base_config, key_index_tuples, rhs)

def _apply_one_cli_override(base_config: dict, override_str: str):
    lhs, rhs = override_str.split('=', 1)  # something.foo.bar=value
    # match anything like name[#][#][#]...
    key_index_tuples = re.findall(r'([^.\[\]]+)([\[\]#0-9]*)\.?', lhs)
    _descend(base_config, key_index_tuples, rhs)
    return base_config


def parse_cli_overrides(base_config: dict, cli_args: list[str]):
    '''Turn CLI overrides like ``foo.bar.baz=1``
    into ``{'foo': {'bar': {'baz': '1'}}}``.

    Repeated keys take the last value seen.

    Overriding a specific list item is possible with subscripting with
    integer indices, e.g. ``foo[1].bar=1`` becomes
    ``foo[1]['bar'] = 1``. Fails when the ``base_config`` doesn't have an
    appropriate-length list (e.g. to set list_key[3] there should be
    pre-existing values for at least items 0, 1, and 2---so it appends at the end)
    '''
    for arg in cli_args:
        _apply_one_cli_override(base_config, arg)

    return base_config

def type_name_or_choices(the_type):
    if dacite.types.is_subclass(the_type, Enum):
        return ", ".join([repr(x.value) for x in the_type])
    elif hasattr(the_type, "__name__"):
        return the_type.__name__
    elif dacite.types.is_union(the_type):
        members = dacite.types.extract_generic(the_type)
        return "(" + ", ".join([type_name_or_choices(m) for m in members]) + ")"
    else:
        return repr(the_type)


def format_field_type(field_type):
    if hasattr(field_type, "__args__"):
        if dacite.types.is_optional(field_type):
            return f"(optional) {format_field_type(field_type.__args__[0])}"
        elif dacite.types.is_union(field_type):
            name = ""
        else:
            name = type_name_or_choices(field_type)
        members = ", ".join([type_name_or_choices(x) for x in field_type.__args__])
        return f"{name}[{members}]"
    return type_name_or_choices(field_type)


def fields(class_or_instance):
    fields = dataclasses.fields(class_or_instance)
    return tuple(f for f in fields if f.init)


def list_fields(cls, prefix="", help_suffix=""):
    for fld in fields(cls):
        for result in list_one_dataclass_field(fld, prefix, help_suffix):
            yield result


def list_one_dataclass_field(fld: dataclasses.Field, prefix, help_suffix):
    name = fld.name
    field_help = fld.metadata.get("help", "")
    fld_type = fld.type
    default = fld.default
    if not isinstance(fld.default_factory, dataclasses._MISSING_TYPE):
        try:
            default = fld.default_factory()
        except TypeError:
            raise TypeError(f"Could not instantiate field {fld} from default factory")
    for result in list_one_field(
        name, fld_type, field_help, prefix, help_suffix, default=default
    ):
        yield result


def list_one_field(
    name, fld_type, field_help, prefix, help_suffix, default=dataclasses._MISSING_TYPE
):
    if not isinstance(default, dataclasses._MISSING_TYPE):
        if isinstance(default, Enum):
            default_value = repr(default.value)
        else:
            default_value = repr(default)
        field_help += f" (default: {default_value})"
    field_help += help_suffix
    field_type_str = format_field_type(fld_type)
    prefixed_name = prefix + name
    yield prefixed_name, f"{field_type_str}", field_help

    if dacite.types.is_union(fld_type) or dacite.types.is_generic_collection(fld_type):
        if dacite.types.is_union(fld_type):
            members = dacite.types.extract_generic(fld_type)
        else:
            members = [fld_type]
        for mtype in members:
            if dataclasses.is_dataclass(mtype):
                for k, v, h in list_fields(
                    mtype,
                    prefix=f"{prefix}{name}.",
                    help_suffix=help_suffix + f" <{mtype.__name__}>",
                ):
                    yield k, v, h
            elif dacite.types.is_generic_collection(mtype):
                orig_type = dacite.types.extract_origin_collection(mtype)
                res = dacite.types.extract_generic(mtype)
                if orig_type is dict:  # mapping
                    collection_key_type, collection_value_type = res
                    if dataclasses.is_dataclass(collection_value_type):
                        for k, v, h in list_fields(
                            collection_value_type,
                            prefix=f"{prefix}{name}.<{collection_key_type.__name__}>.",
                            help_suffix=help_suffix
                            + f" <{format_field_type(collection_value_type)}>",
                        ):
                            yield k, v, h
                    else:
                        pass  # output above with primitive types
                elif orig_type is list:
                    (collection_entry_type,) = res
                    if dataclasses.is_dataclass(collection_entry_type):
                        for k, v, h in list_fields(
                            collection_entry_type,
                            prefix=f"{prefix}{name}[#].",
                            help_suffix=help_suffix
                            + f" <{format_field_type(collection_entry_type)}>",
                        ):
                            yield k, v, h
                    else:
                        pass  # output above with primitive types
                else:
                    pass  # unhandled parameterized generic collection (tuples? sets?)
            else:
                # no need for additional docs for primitive types
                continue
    else:
        if dataclasses.is_dataclass(fld_type):
            for k, v, h in list_fields(fld_type, prefix=f"{prefix}{name}."):
                yield k, v, h
