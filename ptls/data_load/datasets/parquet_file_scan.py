import os
import warnings
from glob import glob
from typing import Union, List
from omegaconf import ListConfig
from sklearn.model_selection import train_test_split


def _scan_path(path):
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        file_path = os.path.join(path, '*.parquet')
        return glob(file_path)
    return []


def train_valid_split(
        data,
        valid_rate: Union[float, int] = None,
        is_sorted: bool =True,
        return_part: str = 'train',
        shuffle_seed: int = 42,
):
    """

    Parameters
    ----------
    data
        objects for split
    valid_rate:
        if set split found files into train-test
        int means valid objects count, float means valid objects rate
    is_sorted:
        sort or not found files. Should be True when `valid_rate` split used
    return_part: one of ['train', 'valid']
        Which part will be returned when `valid_rate` split used
    shuffle_seed:
        random seed for train_test_split
    Returns
    -------
        object list which are the same as `data` when `valid_rate` aren't used or part of `data` if `valid_rate` used
    """
    if is_sorted:
        data = sorted(data)

    if valid_rate is None or valid_rate == 0.0 or valid_rate == 0:
        return data

    if valid_rate is not None and not is_sorted:
        warnings.warn('train_test_split on unsorted data may returns unexpected result. '
                      'Use `is_sorted=True` when `valid_rate > 0.0`')

    train, valid = train_test_split(data, test_size=valid_rate, random_state=shuffle_seed)
    if return_part == 'train':
        return train
    elif return_part == 'valid':
        return valid
    else:
        raise AttributeError('Never happens')


def parquet_file_scan(
        file_path: Union[str, List[str], ListConfig],
        valid_rate: Union[float, int] = None,
        is_sorted: bool =True,
        return_part: str = 'train',
        shuffle_seed: int = 42,
):
    """Scan folder with parquet files and returns file names. Train-valid split possible

    Split should be reproducible with same results when `is_sorted=True` and other parameters don't change.
    This means that you can split files into synchronised train-valid parts with two calls.

    Example:
    >>> files = [1, 2, 3, 4, 6, 5, 9, 8]
    >>> train_files = train_valid_split(files, valid_rate=0.3, return_part='train', shuffle_seed=123)
    >>> valid_files = train_valid_split(files, valid_rate=0.3, return_part='valid', shuffle_seed=123)
    >>> for i in train_files:
    >>>     assert i not in valid_files
    >>> for i in valid_files:
    >>>     assert i not in train_files

    Parameters
    ----------
    file_path:
        path for scan. Can be single file, directory or list of them.
    valid_rate:
        if set split found files into train-test
        int means valid objects count, float means valid objects rate
    is_sorted:
        sort or not found files. Should be True when `valid_rate` split used
    return_part: one of ['train', 'valid']
        Which part will be returned when `valid_rate` split used
    shuffle_seed:
        random seed for train_test_split
    Returns
    -------
        File list which are all found files when `valid_rate` aren't used or part of files if `valid_rate` used

    """
    assert return_part in ('train', 'valid')

    if type(file_path) not in (list, ListConfig):
        file_path = [file_path]

    data_files = []
    for path in file_path:
        data_files.extend(_scan_path(path))

    return train_valid_split(data_files, valid_rate, is_sorted, return_part, shuffle_seed)
