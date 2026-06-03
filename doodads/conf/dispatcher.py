import argparse
import logging
import sys

import textwrap

from .command import Command
from .core import merge_configs, from_dict
from .io import config_to_toml, parse_environment_config, parse_cli_config

__all__ = ("Dispatcher",)


def add_subparser_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-c",
        "--config-file",
        action="append",
        default=[],
        help="Path to config file, repeat to merge multiple",
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Print usage information"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="Dump final configuration state as TOML and exit",
    )
    parser.add_argument(
        "vars", nargs="*", help="Config variables set with 'key.key.key=value' notation"
    )
    return parser


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
    subcommands: list[Command | "Dispatcher"]

    def __init__(self, name, subcommands):
        self.name = name
        self.sub_commands = subcommands

    def configure_logging(self, level):
        logging.basicConfig(level=level)

    def main(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.set_defaults(command_cls=None, command_parser=None)
        self._add_subcommands(parser)
        args = parser.parse_args()
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
        self.configure_logging("DEBUG" if args.verbose else "INFO")
        command = args.command_cls.from_args(args)
        if args.dump_config:
            print(config_to_toml(command))
            sys.exit(0)
        return command._wrap_main()

    @classmethod
    def _add_subcommands(cls: type, parser: argparse.ArgumentParser):
        subps = parser.add_subparsers(
            title="subcommands", description="valid subcommands"
        )
        names_to_subparsers = {}
        for subcommand_cls in cls.subcommands:
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
