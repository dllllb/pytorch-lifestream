import itertools
import logging
import warnings

import numpy as np
from sklearn.metrics import cohen_kappa_score

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


class ListSubset:
    """The same as torch.utils.data.Subset
    """
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
