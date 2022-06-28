import numpy as np
import torch
from collections import defaultdict
from functools import reduce
from ptls.nn import PaddedBatch


class DictTransformer:
    def __init__(self, *args, **kwargs):
        """Mixin constructor
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def is_seq_feature(k: str, x):
        """Check is value sequential feature

        Iterables are:
            np.array
            torch.Tensor

        Not iterable:
            list    - dont supports indexing

        Parameters
        ----------
        k:
            feature_name
        x:
            value for check

        Returns
        -------
            True if value is iterable
        """
        if k == 'event_time':
            return True
        if k.startswith('target'):
            return False
        if type(x) in (np.ndarray, torch.Tensor):
            return True
        return False

    @staticmethod
    def seq_indexing(k: str, x, ix):
        """Apply indexing for seq_features only

        Parameters
        ----------
        k:
            feature name
        x:
            value
        ix:
            indexes

        Returns
        -------

        """
        if DictTransformer.is_seq_feature(k, x):
            return x[ix]
        return x

    @staticmethod
    def get_seq_len(d):
        """Finds a sequence column and return its length

        Parameters
        ----------
        d:
            feature-dict

        """
        if 'event_time' in d:
            return len(d['event_time'])
        return len(next(v for k, v in d.items() if DictTransformer.is_seq_feature(k, v)))


def collate_feature_dict(batch, array_cols=None):
    """Collate feature with arrays to padded batch

    Check feature consistency. Keys for all batch samples should be the same.
    Drops features which are not compatible with torch.Tensor like `str` or `datetime`
    Convert scalar value to tensors like target col

    Parameters
    ----------
    batch:
        list with feature dicts
    array_cols:
        list of columns which are arrays.
        There are `sequences` features and `arrays` which are 1-d array of tensor.
        `sequences` have 1-d shape with different lengths per sample
        `arrays` have any shape that is the same for all samples.
        `sequences` will be padded, `arrays` will be stacked (which is faster)
        Seq lens should be calculated by `sequences` not `arrays`.
        Provide `array_cols` for correct split of array features into `sequences` and `arrays`.

    Returns
    -------
        PaddedBatch
    """
    if array_cols is None:
        array_cols = []

    new_x_ = defaultdict(list)
    for i, x in enumerate(batch):
        for k, v in x.items():
            new_x_[k].append(v)
        assert reduce(
            lambda a, b: ((a[1] is not None and a[1] == b or a[1] is None) and a[0], b),
            map(len, new_x_.values()), (True, None))[0]

    seq_col = next(k for k, v in batch[0].items() if DictTransformer.is_seq_feature(k, v) and k not in array_cols)
    lengths = torch.LongTensor([len(rec[seq_col]) for rec in batch])
    new_x = {}
    for k, v in new_x_.items():
        if type(v[0]) in (np.ndarray, torch.Tensor):
            if k in array_cols or k.startswith('target'):
                new_x[k] = torch.stack(v, dim=0)
            else:
                new_x[k] = torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
        else:
            v = np.array(v)
            if v.dtype.kind == 'i':
                new_x[k] = torch.from_numpy(v).long()
            if v.dtype.kind == 'f':
                new_x[k] = torch.from_numpy(v).float()

    return PaddedBatch(new_x, lengths)
