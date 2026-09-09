from __future__ import annotations
import argparse
import logging
import os
import sys

import textwrap

from .command import Command
from .core import merge_configs, from_dict
from .interface import (
    # config_to_toml,
    parse_environment_config,
    # parse_cli_config,
    list_fields, add_subparser_arguments)

__all__ = ("Dispatcher",)

def print_help(cls, parser):
    print(f"{sys.argv[0]}:")
    parser.print_help()
    print("\nconfiguration keys:")
    fields = list_fields(cls)
    for field_key, field_type, help_text in fields:
        print(f"  {field_key}")
        print(f"      {field_type}")
        if help_text:
            wrapped = textwrap.fill(
                help_text, initial_indent="    ", subsequent_indent="    "
            )
            print(wrapped)


class Dispatcher:
    name: str
    subcommands: list[Command | Dispatcher]

    def __init__(self, name, subcommands):
        self.name = name
        self.subcommands = subcommands

    def parse_args(self, arguments: list[str] | None=None, prog=None):
        parser = argparse.ArgumentParser(add_help=False, prog=prog)
        parser.set_defaults(command_cls=None, command_parser=None)
        self._add_subcommands(parser)
        return parser.parse_args(args=arguments)

    def main(self):
        args = self.parse_args()
        env_var_name_base = args.prog.upper()
        environ = {
            k.replace(f"{env_var_name_base}_", ""): v
            for k, v in os.environ.items()
            if k.startswith(f"{env_var_name_base}_")
        }
        environ_config = parse_environment_config(environ)


        if args.command_cls is None:
            parser.print_help()
            sys.exit(1)
        if args.help:
            print_help(args.command_cls, args.command_parser)
            sys.exit(0)

        command = args.command_cls.from_args(args)
        if args.dump_config:
            # TODO: print(config_to_toml(command))
            sys.exit(0)
        command.configure_logging("DEBUG" if args.verbose else "INFO", base_command_name=self.name)
        return command._wrap_main()

    def _add_subcommands(self, parser: argparse.ArgumentParser):
        subps = parser.add_subparsers(
            title="subcommands", description="valid subcommands"
        )
        names_to_subparsers = {}
        for subcommand_cls in self.subcommands:
            if isinstance(subcommand_cls, Dispatcher):
                subdispatcher_name = subcommand_cls.name
                subp = subps.add_parser(subdispatcher_name, add_help=False)
                subcommand_cls._add_subcommands(subp)
            else:
                if (
                    subcommand_cls.get_name() is None
                    or subcommand_cls.get_name() in names_to_subparsers
                ):
                    raise Exception(f"Invalid command name for {subcommand_cls}")
                subp = subps.add_parser(subcommand_cls.get_name(), add_help=False)
                subp.set_defaults(command_cls=subcommand_cls, command_parser=subp)

            add_subparser_arguments(subp)


def test_nest_subcommands():
    class Foo(Command):
        def main(self):
            pass
    class Bar(Command):
        def main(self):
            pass
    class Baz(Command):
        def main(self):
            pass
    subd = Dispatcher('sub', [
        Foo,
        Bar
    ])
    d = Dispatcher('test', [
        subd,
        Baz,
    ])
    args = d.parse_args(['baz'])
    assert args.command_cls is Baz
    args = d.parse_args(['sub', 'foo'])
    assert args.command_cls is Foo
    args = d.parse_args(['sub', 'bar'])
    assert args.command_cls is Bar
