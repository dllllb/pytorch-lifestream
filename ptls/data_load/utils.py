from functools import partial

import numpy as np
import torch
from collections import defaultdict
from pymonad.maybe import Maybe
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.padded_batch import PaddedBatch
from itertools import compress

torch_to_numpy = torch.from_numpy


def convert_dtype(np_arr):
    np_arr = np.array(np_arr)
    return torch_to_numpy(np_arr, dtype=dtype_dict[np_arr.dtype.kind])


def detect_transform_func(dtype_list):
    transform_func = [partial(torch.nn.utils.rnn.pad_sequence, batch_first=True), torch.stack, None, np.array,
                      convert_dtype]
    return list(compress(transform_func, dtype_list))


def detect_dtype(dict_tup):
    target, tensor = dict_tup[0], dict_tup[1][0]
    is_torch_tensor = isinstance(tensor, torch.Tensor)
    is_torch_tensor_target = all([is_torch_tensor, target.startswith('target')])
    is_np_array = isinstance(tensor, np.ndarray)
    is_list = isinstance(tensor, list)
    is_other = not any([is_list, is_torch_tensor, is_torch_tensor_target, is_np_array])
    return [is_torch_tensor, is_torch_tensor_target, is_np_array, is_list, is_other]


dtype_dict = {'i': torch.long,
              'f': torch.float,
              'b': torch.bool}


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
            new_x[k].append(v)

    new_x = defaultdict(list)
    _ = list(map(_update_dict, enumerate(batch)))
    del _
    seq_col = next(k for k, v in batch[0].items() if FeatureDict.is_seq_feature(v))
    lengths = torch.LongTensor(list(map(partial(_return_len, col_name=seq_col), batch)))
    list_of_transform_func = Maybe.insert(list(new_x.items())).then(
        function=lambda dict_tup: list(map(detect_dtype, dict_tup))). \
        then(function=lambda dtype_list: list(map(detect_transform_func, dtype_list))). \
        maybe(default_value=None, extraction_function=lambda list_of_transform_func: list_of_transform_func)
    for func, dict_tup in zip(list_of_transform_func, new_x.items()):
        new_x[dict_tup[0]] = func[0](dict_tup[1])

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
        return np.hstack((vec[:num - 1], vec[num - 1:].sum()))[:len(vec)]
