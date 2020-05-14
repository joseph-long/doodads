import collections
import os
import os.path
import urllib.request
from urllib.parse import urlparse
import logging
from functools import wraps

log = logging.getLogger(__name__)

__all__ = [
    'supply_argument',
    'PACKAGE_DIR',
    'DATA_DIR',
    'download_path',
    'download',
    'generated_path',
]

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

def download_path(url, filename):
    outpath = os.path.join(DATA_DIR, 'downloads', filename)
    return outpath

def download(url, filename, overwrite=False):
    outpath = download_path(url, filename)
    if overwrite or not os.path.exists(outpath):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        log.info(f'Downloading {url} -> {outpath}')
        urllib.request.urlretrieve(url, outpath)
        log.info(f'Saved in {outpath}')
    return outpath

def download_to(url, filepath):
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        log.info(f'Downloading {url} -> {filepath}')
        urllib.request.urlretrieve(url, filepath)
        log.info(f'Saved in {filepath}')
    return filepath

def generated_path(filename):
    outpath = os.path.join(DATA_DIR, 'generated', filename)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    return outpath

LOADING = object()

class LazyLoadable:
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
        if not super().__getattribute__('_loaded'):
            self._loaded = LOADING  # allow attribute lookup as normal during lazy load
            self._ensure_loaded()
        return super().__getattribute__(name)

RemoteResource = collections.namedtuple('RemoteResource', 'url convert_file_function download_filepath output_filepath')

class RemoteResourceRegistry:
    def __init__(self):
        self.resources = []
    def download_and_convert_all(self):
        for res in self.resources:
            log.info(f'Resource: {res}')
            if not os.path.exists(res.output_filepath):
                log.info(f'output file path {res.output_filepath}')
                if not os.path.exists(res.download_filepath):
                    log.info(f'download {res.download_filepath}')
                    download_to(res.url, res.download_filepath)
                log.info(f'Converting {res.download_filepath} -> {res.output_filepath} with {res.convert_file_function}')
                res.convert_file_function(res.download_filepath, res.output_filepath)
    def add(self, url, convert_file_function, output_filename=None):
        urlparts = urlparse(url)
        download_filename = os.path.basename(urlparts.path)
        download_filepath = download_path(url, download_filename)

        if output_filename is None:
            output_filename = download_filename
        output_filepath = generated_path(output_filename)

        self.resources.append(RemoteResource(
            url=url,
            convert_file_function=convert_file_function,
            download_filepath=download_filepath,
            output_filepath=output_filepath,
        ))
        return output_filepath

REMOTE_RESOURCES = RemoteResourceRegistry()
