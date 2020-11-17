"""
<create_train_loader>: {
    split: {
        type: train-valid
    }
    train: <files_config>
    valid: <files_config>
}

<create_train_loader>: {
    split: {
        type: auto
        by: row|file|hash|date
        valid_size: int|0.0<=float<1.0
        [train_size: int|0.0<=float<1.0]
        sorted: bool
        shuffle_seed: int|None
    }
    data: <files_config>
}

<create_inference_loader>: {
    inference_data: [<files_config>, ...]
}

<files_config>: {
    path: <path>
    file_structure: auto|file|folder|partitioned
    dataset_type: map|iterable
}

"""
from itertools import chain

import torch
from pyhocon.config_parser import ConfigFactory

from tests.dltranz_tests.test_data_load import gen_trx_data


def read_file_gen(file_n):
    rows_in_files = {i: k // 2 for i, k in enumerate(range(10, 30, 1))}  # [5, 5, 6, 6, 7, ..., 13, 14, 14]
    for i, rec in enumerate(gen_trx_data((torch.rand(rows_in_files[file_n]) * 60 + 1).long())):
        rec['uid'] = file_n * 1000 + i
        yield rec


def test_read_file_gen():
    """Self test"""
    data = list(read_file_gen(4))
    assert len(data) == 7
    assert data[2]['uid'] == 4002


def test_loader__files_split_map():
    files = [4, 5, 6, 7]
    data = chain(*(read_file_gen(f) for f in files))
    data = list(data)
    dataset = MapDataset(data)
    dataset = Splitter(dataset)
    dataset = Augmentations(dataset)
    loader = DataLoader(dataset)
    cnt = 0
    for batch in loader:
        cnt += 1
    assert cnt == 1


def test_loader__files_split_iterable():
    files = [4, 5, 6, 7]
    data = chain(*(read_file_gen(f) for f in files))
    dataset = IterableDataset(data)
    dataset = Splitter(dataset)
    dataset = Augmentations(dataset)
    loader = DataLoader(dataset)
    cnt = 0
    for batch in loader:
        cnt += 1
    assert cnt == 1

def test_loader__parts_split_map():
    files = [4, 5, 6, 7]
    data = chain(*(read_file_gen(f) for f in files))
    data = list(data)
    dataset = MapDataset(data)
    dataset = Splitter(dataset)
    dataset = Augmentations(dataset)
    loader = DataLoader(dataset)
    cnt = 0
    for batch in loader:
        cnt += 1
    assert cnt == 1


def test_loader__parts_split_iterable():
    files = [4, 5, 6, 7]
    data = chain(*(read_file_gen(f) for f in files))
    dataset = IterableDataset(data)
    dataset = Splitter(dataset)
    dataset = Augmentations(dataset)
    loader = DataLoader(dataset)
    cnt = 0
    for batch in loader:
        cnt += 1
    assert cnt == 1

