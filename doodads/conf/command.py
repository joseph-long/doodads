import abc
import typing
import dataclasses
import datetime
from datetime import timezone
import logging
import os
import pathlib
import re

from . import interface

T = typing.TypeVar('T')

log = logging.getLogger(__name__)

@dataclasses.dataclass
class AbstractCommand(abc.ABC):
    @abc.abstractmethod
    def configure_logging(self, level, base_command_name: str | None = None):
        ...

    @classmethod
    def get_name(cls):
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
        return name

    @classmethod
    @abc.abstractmethod
    def get_default_config_prefix(cls) -> pathlib.Path:
        ...

    @classmethod
    def get_default_config_path(cls) -> pathlib.Path:
        return cls.get_default_config_prefix() / (cls.get_name() + ".conf")

    def config_to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_config(
        cls,
        *,
        config_path_or_paths: str | list[str] | None=None,
        config_dict: dict|None=None,
        settings_strs: list[str]|None=None,
    ):
        '''Initialize a class instance using config files from disk or a dictionary
        of options

        Parameters
        ----------
        config_path :
        '''

        config_paths = []
        if isinstance(config_path_or_paths, str):
            config_paths.append(config_path_or_paths)
        elif isinstance(config_path_or_paths, list):
            config_paths.extend(config_path_or_paths)

        if settings_strs is None:
            settings_strs = []
        raw_config = _get_config_data(cls.get_default_config_path(), config_paths, settings_strs)
        if config_dict is not None:
            for key, value in config_dict.items():
                if key in raw_config:
                    old_val = raw_config[key]
                    log.info(f"Using provided value {value} for {key} which was set to {old_val} in the loaded config files")
            raw_config.update(config_dict)
        try:
            instance = from_dict(cls, raw_config)
        except (UnexpectedDataError, MissingValueError) as e:
            raise ConfigMismatch(e, raw_config)
        return instance

    @classmethod
    def run(cls):
        parser = argparse.ArgumentParser(add_help=False)
        interface.add_subparser_arguments(parser)
        args = parser.parse_args()
        if args.help:
            print_help(cls, parser)
            sys.exit(0)
        logging.basicConfig(level='WARN')
        logging.getLogger().setLevel('DEBUG' if args.verbose else 'INFO')
        command = cls.from_args(args)
        if args.dump_config:
            print(config_to_toml(command))
            sys.exit(0)
        command._wrap_main()

    @classmethod
    def from_args(cls: type[typing.Self], parsed_args) -> typing.Self:
        try:
            return cls.from_config(config_path_or_paths=parsed_args.config_file, settings_strs=parsed_args.vars)
        except ConfigMismatch as e:
            log.error(f"Applying configuration failed with: {e.original_exception}")
            from pprint import pformat
            log.error(f"File and arguments provided this configuration:\n\n{pformat(e.raw_config)}\n")
            raise

    def _wrap_main(self):
        """Wrap main() to allow `setup` and `teardown` hooks.
        """
        self.setup()
        try:
            self.main()
        except Exception as e:
            log.exception(f"Aborting due to exception: {e}")
        finally:
            self.teardown()


    def setup(self):
        """Runs before main() to perform any checks or setup that must run before `main`
        (e.g. to provide an extension point for a derived class that is in turn a base class
        for others to inherit from, since __post_init__ is tricky)
        """
        pass

    def main(self):
        raise NotImplementedError("Subclasses must implement main()")

    def teardown(self):
        """The post-main counterpart to `setup` that runs in a ``finally:``
        block before exiting
        """
        pass

class Command(AbstractCommand):
    @classmethod
    def get_default_config_prefix(cls) -> pathlib.Path:
        return pathlib.Path('./')

    def configure_logging(self, level, base_command_name: str | None = None):
        if base_command_name:
            name_prefix = f"{base_command_name}_{self.get_name()}"
        else:
            name_prefix = self.get_name()
        run_dir : pathlib.Path = self.get_or_create_run_dir(f"{base_command_name}_{self.get_name()}")
        logging.basicConfig(level=level, filename=run_dir / f"{name_prefix}.log")

class RunDirCommand(AbstractCommand):
    _run_dir : pathlib.Path | None = None
    _previous_run_dir : pathlib.Path | None = None
    _start_time : datetime.datetime | None = None
    destination : pathlib.Path = dataclasses.field(default_factory=lambda: pathlib.Path(os.getcwd()))

    @classmethod
    def get_default_config_prefix(cls) -> pathlib.Path:
        return pathlib.Path('./')

    def _set_previous_run_dir(self, base_name) -> pathlib.Path | None:
        if self._previous_run_dir is not None:
            return self._previous_run_dir

        past_run_dirs = list(sorted(self.destination.glob(f"{base_name}_*/")))
        if len(past_run_dirs):
            self._previous_run_dir = past_run_dirs[-1]
        else:
            self._previous_run_dir = None
        return self._previous_run_dir

    @property
    def run_dir(self) -> pathlib.Path:
        if self._run_dir is None:
            raise AttributeError("No run dir yet, call create_run_dir() first")
        return self._run_dir

    @property
    def previous_run_dir(self) -> pathlib.Path:
        if self._previous_run_dir is None:
            raise AttributeError("No run dir yet, call create_run_dir() first")
        return self._previous_run_dir

    def create_run_dir(self, base_name='run') -> pathlib.Path:
        if self._run_dir is not None:
            return self._run_dir

        self._start_time = datetime.datetime.now(timezone.utc)
        run_timestamp_str = self._start_time.strftime("%Y-%m-%d_%H%M%S")

        self._run_dir = self.destination / f"{base_name}_{run_timestamp_str}"
        self._run_dir.mkdir(exist_ok=True, parents=True)
        self._set_previous_run_dir(base_name)
        return self._run_dir

    def configure_logging(self, level, base_command_name: str | None = None):
        if base_command_name:
            name_prefix = f"{base_command_name}_{self.get_name()}"
        else:
            name_prefix = self.get_name()
        run_dir : pathlib.Path = self.create_run_dir(f"{base_command_name}_{self.get_name()}")
        logging.basicConfig(level=level, filename=run_dir / f"{name_prefix}.log")
