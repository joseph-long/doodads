import os
import os.path
import urllib.request
from urllib.parse import urlparse
import hashlib
from functools import wraps
from appdirs import AppDirs

__all__ = [
    'DIRS',
    'supply_argument',
]

DIRS = AppDirs("doodads", "JLong")

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

def cached_fetch_path(url):
    url_hash = hashlib.md5(url.encode('utf8')).hexdigest()
    urlparts = urlparse(url)
    outpath = os.path.join(DIRS.user_cache_dir, url_hash, os.path.basename(urlparts.path))
    return outpath

def cached_fetch(url, force=False):
    outpath = cached_fetch_path(url)
    if force or not os.path.exists(outpath):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        urllib.request.urlretrieve(url, outpath)
    return outpath

def cache_path(filename):
    return os.path.join(DIRS.user_cache_dir, filename)
