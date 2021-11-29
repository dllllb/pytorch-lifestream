import argparse
import itertools
import logging
import os
import pickle
import subprocess
import sys
from glob import glob

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score

import warnings
import functools


logger = logging.getLogger(__name__)


def block_iterator(iterator, size):
    bucket = list()
    for e in iterator:
        bucket.append(e)
        if len(bucket) >= size:
            yield bucket
            bucket = list()
    if bucket:
        yield bucket


def cycle_block_iterator(iterator, size):
    return block_iterator(itertools.cycle(iterator), size)


def get_cls(cls_name):
    i = cls_name.split('.')
    mod = __import__('.'.join(i[:-1]), fromlist=[i[-1]])
    cls = getattr(mod, i[-1])
    if cls is None:
        raise AttributeError(f'Unknown class name: "{cls_name}"')
    return cls


class ListSubset:
    def __init__(self, delegate, idx_to_take):
        self.delegate = delegate
        self.idx_to_take = idx_to_take

    def __len__(self):
        return len(self.idx_to_take)

    def __getitem__(self, idx):
        return self.delegate[self.idx_to_take[idx]]

    def __iter__(self):
        for i in self.idx_to_take:
            yield self.delegate[i]


def init_logger(name, level=logging.INFO, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode='w')
        formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_conf(args=None):
    import sys
    import os
    from pyhocon import ConfigFactory

    p = argparse.ArgumentParser()
    p.add_argument('-c', '--conf', nargs='+')
    args, overrides = p.parse_known_args(args)

    logger.info(f'args: {args}, overrides: {overrides}')

    init_conf = f"script_path={os.path.dirname(os.path.abspath(sys.argv[0]))}"
    file_conf = ConfigFactory.parse_string(init_conf)

    if args is not None and args.conf is not None:
        for name in args.conf:
            logger.info(f'Load config from "{name}"')
            file_conf = ConfigFactory.parse_file(name, resolve=False).with_fallback(file_conf, resolve=False)

    overrides = ','.join(overrides)
    over_conf = ConfigFactory.parse_string(overrides)
    if len(over_conf) > 0:
        logger.info(f'New overrides:')

        def print_differences(root=''):
            if len(root) > 0:
                c = over_conf[root[:-1]]
            else:
                c = over_conf

            for k, v in c.items():
                old = file_conf.get(f"{root}{k}", None)
                if isinstance(v, dict) and isinstance(old, dict):
                    print_differences(f'{root}{k}.')
                else:
                    logger.info(f'    For key "{root}{k}" provided new value "{v}", was "{old}"')

        print_differences()
    conf = over_conf.with_fallback(file_conf)
    return conf


def config_coalesce(conf, *keys, default=None, raise_if_missing=False):
    from pyhocon import ConfigMissingException
    for k in keys:
        try:
            v = conf[k]
            return v
        except ConfigMissingException:
            pass
    if raise_if_missing:
        raise ConfigMissingException(f'These keys are not found: [{keys}]')
    return default


def get_data_files(params):
    path_wc = params['path_wc']

    if 'data_path' in params:
        path_wc = os.path.join(params['data_path'], path_wc)

    files = glob(path_wc)
    logger.info(f'Found {len(files)} files in "{path_wc}"')

    max_files = params.get('max_files', None)

    if max_files is not None:
        if type(max_files) is int:
            files = files[:max_files]
            logger.info(f'First {len(files)} files are available')
        elif type(max_files) is float:
            max_files = int(max_files * len(files))
            files = files[:max_files]
            logger.info(f'First {len(files)} files are available')
        else:
            raise AttributeError(f'Wrong type of `dataset.max_files`: {type(max_files)}')
    else:
        logger.info(f'All {len(files)} files are available')
    return sorted(files)


def _read_parquet(file_name):
    py_script = f"""
import pandas as pd
import pickle
import sys

df = pd.read_parquet('{file_name}')
sys.stdout.buffer.write(pickle.dumps(df))
    """
    p = subprocess.Popen([sys.executable, '-c', py_script], stdout=subprocess.PIPE)
    data, _ = p.communicate()
    return pickle.loads(data)


def _write_parquet(df, file_name):
    py_script = f"""
import pandas as pd
import pickle
import sys

df = pickle.loads(sys.stdin.buffer.read())
df.to_parquet('{file_name}')
    """
    p = subprocess.Popen([sys.executable, '-c', py_script], stdin=subprocess.PIPE)
    p.communicate(pickle.dumps(df))


def read_pandas(file_name):
    ext = os.path.splitext(file_name)[1]
    if ext == '.csv':
        return pd.read_csv(file_name)
    elif ext == '.pickle':
        return pd.read_pickle(file_name)
    elif ext == '.parquet':
        return _read_parquet(file_name)
    else:
        raise NotImplementedError(f'Unknown file extension: {ext}')


def write_pandas(df, file_name):
    ext = os.path.splitext(file_name)[1]
    if ext == '.csv':
        df.to_csv(file_name, sep=',', header=True, index=False)
    elif ext == '.pickle':
        df.to_pickle(file_name)
    elif ext == '.parquet':
        return _write_parquet(df, file_name)
    else:
        raise NotImplementedError(f'Unknown file extension: {ext}')


def plot_arrays(a, b, title=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import patches

    def plot_a(x, top_s, offset, **params):
        plt.plot((x, x), (top_s * -0.05, top_s * offset), alpha=0.6, **params)
        plt.text(x, top_s * offset * 1.25, str(x))

    x_min = min(a + b)
    x_max = max(a + b)

    x_len = x_max - x_min

    x_min -= x_len * 0.05
    x_max += x_len * 0.05

    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color(None)
    ax.spines['left'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.yaxis.set_visible(False)

    plt.xlim(x_min, x_max)
    plt.ylim(-0.3, 0.3)
    for x in a:
        plot_a(x, 1, offset=0.2, color='darkblue', linestyle='-', linewidth=4)
    plot_a(np.mean(a).round(5), 1, offset=0.14, color='darkblue', linestyle=':', linewidth=2)
    for x in b:
        plot_a(x, -1, offset=0.2, color='darkgreen', linestyle='-', linewidth=4)
    plot_a(np.mean(b).round(5), -1, offset=0.14, color='darkgreen', linestyle=':', linewidth=2)

    if len(a) >= 3:
        _mean = np.mean(a)
        _std = np.std(a)
        rect = patches.Rectangle((_mean - 2 * _std, -0.025), _std * 4, 0.12, color='darkblue', alpha=0.2)
        ax.add_patch(rect)
    if len(b) >= 3:
        _mean = np.mean(b)
        _std = np.std(b)
        rect = patches.Rectangle((_mean - 2 * _std, 0.025), _std * 4, -0.12, color='darkgreen', alpha=0.2)
        ax.add_patch(rect)

    if title:
        plt.title(title)


def eval_kappa_regression(y_true, y_pred):
    dist = {3: 0.5, 0: 0.239, 2: 0.125, 1: 0.136} # ground shares
    
    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)

    y_pred = np.where(
        y_pred<=bound[0],
        0,
        np.where(
            y_pred<=bound[1],
            1,
            np.where(
                y_pred<=bound[2],
                2,
                3
            )
        )
    ).reshape(y_true.shape)

    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def switch_reproducibility_on():
    import torch
    import random

    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class Deprecated:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def __init__(self, message):
        self._message = message

    def __call__(self, func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn(f"Call to deprecated function {func.__name__}. {self._message}",
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)
        return new_func
