import functools
import numpy as np
import operator
import random
import torch

from collections import defaultdict
from torch.utils.data import Dataset

from dltranz.seq_encoder import PaddedBatch


class SOPModel(torch.nn.Module):
    def __init__(self, base_model, embeds_dim, config):
        super().__init__()
        self.base_model = base_model
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embeds_dim * 2, config['hidden_size'], bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(config['drop_p']),
            torch.nn.BatchNorm1d(config['hidden_size']),
            torch.nn.Linear(config['hidden_size'], 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        left, right = x
        x = torch.cat([self.base_model(left), self.base_model(right)], dim=1)
        return self.head(x).squeeze(-1)


class SOPDataset(Dataset):
    def __init__(self, delegate):
        self.delegate = delegate

    def __len__(self):
        return len(self.delegate)

    def __iter__(self):
        for rec in iter(self.delegate):
            yield self._one_item(rec)

    def __getitem__(self, idx):
        item = self.delegate[idx]
        if type(item) is list:
            return [self._one_item(t) for t in item]
        else:
            return self._one_item(item)

    def _one_item(self, item):
        length = len(next(iter(item.values())))
        l_length = random.randint(length//4, 3*length//4)
        left = {k: v[:l_length] for k, v in item.items()}
        right = {k: v[l_length:] for k, v in item.items()}

        target = random.randint(0, 1)
        return ((left, right), target) if target else ((right, left), target)


class ConvertingTrxDataset(Dataset):
    def __init__(self, delegate, style='map'):
        self.delegate = delegate
        if hasattr(delegate, 'style'):
            self.style = delegate.style
        else:
            self.style = style

    def __len__(self):
        return len(self.delegate)

    def __iter__(self):
        for rec in iter(self.delegate):
            yield self._one_item(rec)

    def __getitem__(self, idx):
        item = self.delegate[idx]
        if type(item) is list:
            return [self._one_item(t) for t in item]
        else:
            return self._one_item(item)

    def _one_item(self, x):
        x = {k: torch.from_numpy(self.to_torch_compatible(v)) for k, v in x.items()}
        return x

    @staticmethod
    def to_torch_compatible(a):
        if a.dtype == np.int8:
            return a.astype(np.int16)
        return a


def padded_collate(batch):
    new_x_ = defaultdict(list)
    for x in batch:
        for k, v in x.items():
            new_x_[k].append(v)

    lengths = torch.IntTensor([len(e) for e in next(iter(new_x_.values()))])
    new_x = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x_.items()}
    return PaddedBatch(new_x, lengths)


def collate_splitted_pairs(batch):
    batch = functools.reduce(operator.iadd, batch)
    return (
        (
            padded_collate([left for (left, _), _ in batch]),
            padded_collate([right for (_, right), _ in batch])
        ),
        torch.tensor([y for _, y in batch])
    )
