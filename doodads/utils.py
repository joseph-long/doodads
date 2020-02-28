import os
import os.path
import urllib.request
from urllib.parse import urlparse
import hashlib
from functools import wraps

__all__ = [
    'supply_argument',
    'PACKAGE_DIR',
    'DATA_DIR',
    'download_path',
    'download',
    'generated_path',
]

PACKAGE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PACKAGE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

def supply_argument(**override_kwargs):
    '''
    Decorator to supply a keyword argument using a callable if
    it is not provided.
    '''

    def decorator(f):
        @wraps(f)
        def inner(*args, **kwargs):
            for kwarg in override_kwargs:
                if kwarg not in kwargs:
                    kwargs[kwarg] = override_kwargs[kwarg]()
            return f(*args, **kwargs)
        return inner
    return decorator

def download_path(url, filename):
    outpath = os.path.join(DATA_DIR, 'downloads', filename)
    return outpath

def download(url, filename, overwrite=False):
    outpath = download_path(url, filename)
    if overwrite or not os.path.exists(outpath):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        urllib.request.urlretrieve(url, outpath)
    return outpath

def generated_path(filename):
    return os.path.join(DATA_DIR, 'generated', filename)
