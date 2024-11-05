from collections import defaultdict
from functools import partial, reduce

import numpy as np
import torch

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.padded_batch import PaddedBatch

torch_to_numpy = torch.from_numpy

transform_func = {'seq_tensor': partial(torch.nn.utils.rnn.pad_sequence,
                                        batch_first=True),
                  'target_tensor': torch.stack}


def detect_transform_func(dict_tup):
    target, tensor = dict_tup[0], dict_tup[1][0]
    tensor_type = 'target_tensor' if all([isinstance(tensor, torch.Tensor), target.startswith('target')]) \
        else 'seq_tensor'
    return transform_func[tensor_type]


# def _collate_feature_dict(batch):
#     """Collate feature with arrays to padded batch.
#     Check feature consistency. Keys for all batch samples should be the same.
#     Convert scalar value to tensors like target col.
#
#     Args:
#         batch: list with feature dicts
#     Returns:
#         PaddedBatch
#     """
#
#     def _return_len(record, col_name):
#         return len(record[col_name])
#
#     def _update_dict(batch_tuple):
#         batch_iter = iter(batch_tuple[1].items())
#         for k, v in batch_iter:
#             if any([k.__contains__('time'), k.__contains__('date')]):
#                 dtype = TORCH_DATETIME_DTYPE
#             elif k.__contains__('group'):
#                 dtype = TORCH_GROUP_DTYPE
#             elif isinstance(v, torch.Tensor):
#                 dtype = v.dtype
#             # if v is an int, float, or bool, convert it to a tensor
#             elif isinstance(v, (int, float, bool)):
#                 dtype = TORCH_EMB_DTYPE
#                 v = torch.tensor([v], dtype=dtype)
#             else:
#                 dtype = TORCH_EMB_DTYPE
#             v = v.type(dtype)
#             new_x[k].append(v)
#
#     new_x = defaultdict(list)
#     _ = list(map(_update_dict, enumerate(batch)))
#     del _
#     seq_col = next(k for k, v in batch[0].items() if FeatureDict.is_seq_feature(k, v))
#     lengths = torch.LongTensor(list(map(partial(_return_len, col_name=seq_col), batch)))
#     list_of_transform_func = Maybe.insert(iter(new_x.items())). \
#         maybe(default_value=None, extraction_function=lambda dict_tup: list(map(detect_transform_func, dict_tup)))
#
#     collated = {dict_tup[0]: list_of_transform_func[idx](dict_tup[1]) for idx, dict_tup in enumerate(new_x.items())}
#
#     return PaddedBatch(collated, lengths)


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
        if isinstance(v[0], torch.Tensor):
            if k.startswith('target'):
                new_x[k] = torch.stack(v, dim=0)
            else:
                new_x[k] = torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
        elif isinstance(v[0], np.ndarray):
            new_x[k] = v  # list of arrays[object]
        elif isinstance(v[0], list):
            new_x[k] = np.array(v, dtype=object)
        else:
            v = np.array(v)
            if v.dtype.kind == 'i':
                new_x[k] = torch.from_numpy(v).long()
            elif v.dtype.kind == 'f':
                new_x[k] = torch.from_numpy(v).float()
            elif v.dtype.kind == 'b':
                new_x[k] = torch.from_numpy(v).bool()
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
        return np.hstack((vec[:num - 1], vec[num - 1:].sum()))[:len(vec)]


def collate_multimodal_feature_dict(batch):
    res = {}
    for source, source_batch in batch.items():
        res[source] = collate_feature_dict(source_batch)
    return res


def get_dict_class_labels(batch):
    res = defaultdict(list)
    for i, samples in enumerate(batch):
        for source, values in samples.items():
            for _ in values:
                res[source].append(i)
    for source in res:
        res[source] = torch.LongTensor(res[source])
    return dict(res)


def init_worker(cls):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # single-process data loading, return the full iterator
        cls._worker_id = 0
        cls._num_workers = 1
        cls._shuffle_seed = cls.shuffle_seed
    else:  # in a worker process
        cls._worker_id = worker_info.id
        cls._num_workers = worker_info.num_workers
        cls._shuffle_seed = worker_info.seed
