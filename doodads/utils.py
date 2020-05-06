import os
import os.path
import urllib.request
import logging
from functools import wraps

logger = logging.getLogger(__name__)

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
        logger.info(f'Downloading {url} -> {outpath}')
        urllib.request.urlretrieve(url, outpath)
        logger.info(f'Saved in {outpath}')
    return outpath

def generated_path(filename):
    outpath = os.path.join(DATA_DIR, 'generated', filename)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    return outpath

LOADING = object()

class LazyLoadable:
    def __init__(self, file_path):
        self.file_path = file_path
        self._loaded = False
    @property
    def exists(self):
        return os.path.exists(self.file_path)
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
