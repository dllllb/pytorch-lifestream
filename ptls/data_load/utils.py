from functools import partial

import numpy as np
import torch
from collections import defaultdict
from pymonad.maybe import Maybe
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.padded_batch import PaddedBatch
from ptls.constant_repository import TORCH_EMB_DTYPE, TORCH_DATETIME_DTYPE, TORCH_GROUP_DTYPE

torch_to_numpy = torch.from_numpy

transform_func = {'seq_tensor': partial(torch.nn.utils.rnn.pad_sequence, batch_first=True),
                  'target_tensor': torch.stack}


def detect_transform_func(dict_tup):
    target, tensor = dict_tup[0], dict_tup[1][0]
    tensor_type = 'target_tensor' if all([isinstance(tensor, torch.Tensor), target.startswith('target')]) \
        else 'seq_tensor'
    return transform_func[tensor_type]


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

    def _return_len(record, col_name):
        return len(record[col_name])

    def _update_dict(batch_tuple):
        batch_iter = iter(batch_tuple[1].items())
        for k, v in batch_iter:
            if any([k.__contains__('time'), k.__contains__('date')]):
                dtype = TORCH_DATETIME_DTYPE
            elif k.__contains__('group'):
                dtype = TORCH_GROUP_DTYPE
            else:
                dtype = TORCH_EMB_DTYPE
            v = v.type(dtype)
            new_x[k].append(v)

    new_x = defaultdict(list)
    _ = list(map(_update_dict, enumerate(batch)))
    del _
    seq_col = next(k for k, v in batch[0].items() if FeatureDict.is_seq_feature(v))
    lengths = torch.LongTensor(list(map(partial(_return_len, col_name=seq_col), batch)))
    list_of_transform_func = Maybe.insert(iter(new_x.items())). \
        maybe(default_value=None, extraction_function=lambda dict_tup: list(map(detect_transform_func, dict_tup)))

    collated = {dict_tup[0]: list_of_transform_func[idx](dict_tup[1]) for idx, dict_tup in enumerate(new_x.items())}

    return PaddedBatch(collated, lengths)


def collate_target(x, num=1):
    vec = np.array(x, dtype=np.float32)
    if num == 1:
        return vec.sum()
    elif abs(num) >= len(vec):
        return vec
    elif num < 0:
        return vec[:abs(num)]
    else:
        return np.hstack((vec[:num - 1], vec[num - 1:].sum()))[:len(vec)]
