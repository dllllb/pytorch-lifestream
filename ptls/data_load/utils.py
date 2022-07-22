import numpy as np
import torch
from collections import defaultdict
from functools import reduce

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.padded_batch import PaddedBatch


def collate_feature_dict(batch):
    """Collate feature with arrays to padded batch

    Check feature consistency. Keys for all batch samples should be the same.
    Convert scalar value to tensors like target col

    Parameters
    ----------
    batch:
        list with feature dicts
    Returns
    -------
        PaddedBatch
    """
    new_x_ = defaultdict(list)
    for i, x in enumerate(batch):
        for k, v in x.items():
            new_x_[k].append(v)
        assert reduce(
            lambda a, b: ((a[1] is not None and a[1] == b or a[1] is None) and a[0], b),
            map(len, new_x_.values()), (True, None))[0]

    seq_col = next(k for k, v in batch[0].items() if FeatureDict.is_seq_feature(k, v))
    lengths = torch.LongTensor([len(rec[seq_col]) for rec in batch])
    new_x = {}
    for k, v in new_x_.items():
        if type(v[0]) is torch.Tensor:
            if k.startswith('target'):
                new_x[k] = torch.stack(v, dim=0)
            else:
                new_x[k] = torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
        elif type(v[0]) is np.ndarray:
            new_x[k] = v  # list of arrays[object]
        else:
            v = np.array(v)
            if v.dtype.kind == 'i':
                new_x[k] = torch.from_numpy(v).long()
            elif v.dtype.kind == 'f':
                new_x[k] = torch.from_numpy(v).float()
            else:
                new_x[k] = v

    return PaddedBatch(new_x, lengths)


def collate_target(x, num=1):
    vec = np.array(x, dtype=np.float32)
    if num == 1:
        return vec.sum()
    elif abs(num) >= len(vec):
        return vec
    elif num < 0:
        return vec[:abs(num)]
    else:
        return np.hstack((vec[:num-1], vec[num-1:].sum()))[:len(vec)]
