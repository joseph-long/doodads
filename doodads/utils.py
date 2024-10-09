import time
import collections
import hashlib
import os
import os.path
import time
import urllib.request
from urllib.parse import urlparse
import logging
import requests
import numpy as np
from numpy.lib.recfunctions import append_fields
from functools import wraps, partial
import typing
import astropy.units as u

log = logging.getLogger(__name__)

__all__ = [
    'YellingProxy',
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
    'ArrayOrQuantity',
    'is_scalar',
    'append_fields',
    'display_all',
    'measure_timing',
]


ArrayOrQuantity = typing.Union[np.ndarray, u.Quantity]

class YellingProxy:
    def __init__(self, message):
        self.message = message

    def __getattr__(self, name: str):
        raise AttributeError(f"trying to access attribute '{name}': {self.message}")


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

def retrieve_and_store(url, filepath):
    # NOTE the stream=True parameter below
    requests.packages.urllib3.disable_warnings(category=requests.packages.urllib3.exceptions.InsecureRequestWarning)
    with requests.get(url, stream=True, verify=False) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return filepath

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
                retrieve_and_store(url, filepath)
                break
            except urllib.error.URLError as e:
                attempt_num = DOWNLOAD_RETRIES - retries
                log.warn(f'(attempt {attempt_num} / {DOWNLOAD_RETRIES}) Failed to retrieve {url}, exception was {e}')
                retries -= 1
                if retries > 0:
                    log.warn(f'Sleeping for two seconds before retrying...')
                    time.sleep(2)
                else:
                    try:
                        # Clean up any partial download
                        os.remove(filepath)
                    except FileNotFoundError:
                        # but don't worry if there isn't one
                        pass
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
    filepath = None
    _loaded = False
    _lazy_attr_allowlist = (
        'filepath',
        'exists',
        '_loaded',
        '_ensure_loaded',
        '_lazy_load',
        '__dict__',
        '_rehydrate',
    )
    def __init__(self, filepath):
        self.filepath = filepath
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
        if name in super().__getattribute__('_lazy_attr_allowlist'):
            return super().__getattribute__(name)
        if not self._loaded:
            self._loaded = LOADING  # allow attribute lookup as normal during lazy load
            self._ensure_loaded()
        return super().__getattribute__(name)

    # pickle support
    @classmethod
    def _rehydrate(cls, serialized_contents):
        instance = object.__new__(cls)
        instance.__dict__.update(serialized_contents)
        return instance

    def __reduce__(self):
        self._ensure_loaded()
        return self._rehydrate, (self.__dict__,)

class BaseResource:
    def __init__(self, converter_function, output_filename):
        self.converter_function = converter_function
        self.output_filepath = generated_path(output_filename)
    @property
    def exists(self):
        return os.path.exists(self.output_filepath)
    def ensure_exists(self):
        raise NotImplementedError("Subclasses must implement ensure_exists()")
    def convert(self):
        raise NotImplementedError("Subclasses must implement convert()")
    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.output_filepath}>'

class CustomRemoteResource(BaseResource):
    """Remote resource that is generated by an arbitrary
    function taking the output_filepath as an arg, bypassing the
    download cache
    """
    def ensure_exists(self):
        if not self.exists:
            self.convert()
    def convert(self):
        self.converter_function(self.output_filepath)

class RemoteResource(BaseResource):
    '''Represent a single resource at `url` converted into a file at
    `output_filepath` by `converter_function` which takes
    the path to the download on disk and the output path as arguments
    '''
    def __init__(self, *, url, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        urlparts = urlparse(url)
        download_filename = os.path.basename(urlparts.path)
        self.download_filepath = unique_download_path(url, download_filename)
    def convert(self):
        self.converter_function(self.download_filepath, self.output_filepath)
    def ensure_exists(self):
        if not self.exists:
            log.info(f'Resource not yet generated at output file path {self.output_filepath}')
            self.download()
            log.info(f'Converting {self.url} -> {self.output_filepath} with {self.converter_function}')
            self.convert()
        else:
            log.info(f'{self.download_filepath} -> {self.output_filepath} exists')
    def download(self):
        download_to(self.url, self.download_filepath)
    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.url}>'

class CollectionResource(BaseResource):
    def __init__(self, resources, converter_function, output_filename):
        super().__init__(converter_function, output_filename)
        self.resources = resources
    def convert(self):
        for res in self.resources:
            res.ensure_exists()
        self.converter_function([res.output_filepath for res in self.resources], self.output_filepath)
    def ensure_exists(self):
        if not self.exists:
            self.convert()

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
            log.info(f"Processing {len(resource_list)} resources for module {mod}")
            for res in resource_list:
                res.ensure_exists()

    def add_from_url(self, module, url, converter_function=os.symlink, output_filename=None):
        if output_filename is None:
            urlparts = urlparse(url)
            download_filename = os.path.basename(urlparts.path)
            output_filename = download_filename

        res = RemoteResource(
            url=url,
            converter_function=converter_function,
            output_filename=output_filename,
        )
        return self.add(module, res)

    def add(self, module, res):
        self.resources[module].append(res)
        return res

REMOTE_RESOURCES = RemoteResourceRegistry()

class TaskRegistry:
    def __init__(self):
        self.tasks = []
    def run_all(self):
        import matplotlib
        matplotlib.use('Agg')
        from astropy.visualization import quantity_support
        quantity_support()
        for t in self.tasks:
            t()
    def add(self, task_func, *func_args, **func_kwargs):
        self.tasks.append(partial(task_func, *func_args, **func_kwargs))

DIAGNOSTICS = TaskRegistry()

def can_be_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def read_webplotdigitizer(file_handle):
    headers = None
    datasets = None
    for line in file_handle:
        if headers is None:
            headers = line.split(',')[::2]
            datasets = {name: {'x': [], 'y': []} for name in headers}
        else:
            assert headers is not None
            bits = line.split(',')
            if not any(map(can_be_float, bits)):
                continue
            for idx, datum in enumerate(bits):
                if not datum.strip():
                    continue
                is_x = True if idx % 2 == 0 else False
                name = headers[idx // 2]
                dataset = datasets[name]
                if is_x:
                    dataset['x'].append(float(datum))
                else:
                    dataset['y'].append(float(datum))
    for name in headers:
        datasets[name]['x'] = np.asarray(datasets[name]['x'])
        datasets[name]['y'] = np.asarray(datasets[name]['y'])
        idxs_to_sort = np.argsort(datasets[name]['x'])
        datasets[name]['x'] = datasets[name]['x'][idxs_to_sort]
        datasets[name]['y'] = datasets[name]['y'][idxs_to_sort]
    return datasets

def is_scalar(val):
    # n.b. Quantity objects don't behave with np.isscalar, but have .isscalar attributes
    # so we check for those first, falling back to np.isscalar
    if isinstance(val, u.Quantity):
        is_scalar = val.isscalar
    elif np.isscalar(val):
        is_scalar = True
    else:
        is_scalar = False
    return is_scalar


def convert_obj_cols_to_str(arr):
    from numpy.lib.recfunctions import drop_fields, append_fields
    names = []
    cols = []
    for name, dtype in arr.dtype.fields.items():
        if dtype[0] == np.dtype('O'):
            names.append(name)
            cols.append(arr[name].astype(str))
    if len(names):
        arr = drop_fields(arr, names)
        arr = append_fields(arr, names, cols)
    return arr

def display_all(arg):
    from IPython.display import display
    import pandas as pd
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(arg)

def measure_timing(func, trials=1, warmups=0):
    '''Run `func` `warmups` times as a warmup then
    measure timing in nanoseconds for `trials` trials

    Parameters
    ----------
    func : callable
    trials : int
    warmups : int

    Returns
    -------
    delta_ns : ndarray
    '''
    for i in range(warmups):
        func()
    delta_ns = []
    for i in range(trials):
        nanos_start = time.perf_counter_ns()
        func()
        nanos_end = time.perf_counter_ns()
        delta_ns.append(nanos_end - nanos_start)
    return np.squeeze(delta_ns)
