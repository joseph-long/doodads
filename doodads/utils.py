import collections
import hashlib
import os
import os.path
import time
import urllib.request
from urllib.parse import urlparse
import logging
from functools import wraps, partial

log = logging.getLogger(__name__)

__all__ = [
    'LazyLoadable',
    'RemoteResource',
    'RemoteResourceRegistry',
    'TaskRegistry',
    'download_to',
    'generated_path',
    'supply_argument',
    'unique_download_path',
    'PACKAGE_DIR',
    'DATA_DIR',
    'DIAGNOSTICS',
    'REMOTE_RESOURCES',
]

DOWNLOAD_RETRIES = 3

PACKAGE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.expanduser('~/.local/share/doodads/'), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

def supply_argument(**override_kwargs):
    '''
    Decorator to supply a keyword argument using a callable if
    it is not provided.
    '''

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            for kwarg in override_kwargs:
                if kwarg not in kwargs:
                    kwargs[kwarg] = override_kwargs[kwarg]()
            return func(*args, **kwargs)
        return inner
    return decorator

def unique_download_path(url, filename):
    hasher = hashlib.sha256()
    hasher.update(url.encode('utf8'))
    outpath = os.path.join(DATA_DIR, 'downloads', hasher.hexdigest(), filename)
    return outpath

def download_to(url, filepath_or_dirpath, overwrite=False):
    '''Download a URL to a destination directory or filename

    Parameters
    ----------
    url : str
    filepath_or_dirpath : str
    overwrite : bool
    '''
    path_part, file_part = os.path.split(filepath_or_dirpath)
    if os.path.isdir(filepath_or_dirpath) or file_part == '':
        dirpath = filepath_or_dirpath
        urlparts = urlparse(url)
        download_filename = os.path.basename(urlparts.path)
        filepath = os.path.join(dirpath, download_filename)
    else:
        dirpath = path_part
        filepath = filepath_or_dirpath
    os.makedirs(dirpath, exist_ok=True)
    if overwrite or not os.path.exists(filepath):
        log.info(f'Downloading {url} -> {filepath}')
        retries = DOWNLOAD_RETRIES
        while retries > 0:
            try:
                urllib.request.urlretrieve(url, filepath)
            except urllib.error.URLError as e:
                attempt_num = DOWNLOAD_RETRIES - retries
                log.warn(f'(attempt {attempt_num} / {DOWNLOAD_RETRIES}) Failed to retrieve {url}, exception was {e}')
                retries -= 1
                if retries > 0:
                    log.warn(f'Sleeping for two seconds before retrying...')
                    time.sleep(2)
                else:
                    raise

    else:
        log.info(f'Existing download for {url} -> {filepath}, pass overwrite=True to replace')
    return filepath

def generated_path(filename):
    outpath = os.path.join(DATA_DIR, 'generated', filename)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    return outpath

LOADING = object()

class LazyLoadable:
    _lazy_attr_whitelist = (
        'filepath',
        'exists',
        '_loaded',
        '_ensure_loaded',
        '_lazy_load',
    )
    def __init__(self, filepath):
        self.filepath = filepath
        self._loaded = False
    @property
    def exists(self):
        return os.path.exists(self.filepath)
    def _ensure_loaded(self):
        if self._loaded in (False, LOADING):
            self._lazy_load()
            self._loaded = True
    def _lazy_load(self):
        raise NotImplementedError("Subclasses must implement _lazy_load")
    def __getattribute__(self, name):
        if name in super().__getattribute__('_lazy_attr_whitelist'):
            return super().__getattribute__(name)
        if not self._loaded:
            self._loaded = LOADING  # allow attribute lookup as normal during lazy load
            self._ensure_loaded()
        return super().__getattribute__(name)

class RemoteResource:
    '''Represent a single resource at `url` converted into a file at
    `output_filepath` by `converter_function` which takes
    the path to the download on disk and the output path as arguments
    '''
    def __init__(self, url, converter_function, output_filepath):
        self.url = url
        urlparts = urlparse(url)
        download_filename = os.path.basename(urlparts.path)
        self.download_filepath = unique_download_path(url, download_filename)
        self.converter_function = converter_function
        self.output_filepath = output_filepath
    @property
    def exists(self):
        return os.path.exists(self.output_filepath)
    def convert(self):
        self.converter_function(self.download_filepath, self.output_filepath)
    def download(self):
        download_to(self.url, self.download_filepath)
    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.url}>'

class RemoteResourceRegistry:
    def __init__(self, resources=None):
        self.resources = collections.defaultdict(list) if resources is None else resources
    def filter(self, exclude):
        if exclude is None:
            exclude = []
        return self.__class__(resources={
            k: v
            for k, v
            in self.resources.items()
            if k not in exclude
        })
    def download_and_convert(self):
        for mod, resource_list in self.resources.items():
            for res in resource_list:
                if not res.exists:
                    log.info(f'Resource for {mod} not yet generated at output file path {res.output_filepath}')
                    res.download()
                    log.info(f'Converting {res.url} -> {res.output_filepath} with {res.converter_function}')
                    res.convert()
    def add(self, module, url, converter_function, output_filename=None):
        if output_filename is None:
            urlparts = urlparse(url)
            download_filename = os.path.basename(urlparts.path)
            output_filename = download_filename
        output_filepath = generated_path(output_filename)

        res = RemoteResource(
            url=url,
            converter_function=converter_function,
            output_filepath=output_filepath,
        )
        self.resources[module].append(res)
        return res

REMOTE_RESOURCES = RemoteResourceRegistry()

class TaskRegistry:
    def __init__(self):
        self.tasks = []
    def run_all(self):
        for t in self.tasks:
            t()
    def add(self, task_func, *func_args, **func_kwargs):
        self.tasks.append(partial(task_func, *func_args, **func_kwargs))

DIAGNOSTICS = TaskRegistry()
