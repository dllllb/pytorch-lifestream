import functools
import operator
import random
import torch

from torch.utils.data import Dataset

from dltranz.sop import padded_collate


class NSPDataset(Dataset):
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

        return left, right


def collate_nsp_pairs(batch):
    batch = functools.reduce(operator.iadd, batch)

    lefts = [left for (left, _), _ in batch] * 2

    rights = [right for (_, right), _ in batch]
    rights_ = rights[:]
    random.shuffle(rights_)
    rights += rights_

    targets = torch.cat([
        torch.ones(len(rights_), dtype=torch.int64),
        torch.zeros(len(rights_), dtype=torch.int64),
    ])

    return (
        (
            padded_collate(lefts),
            padded_collate(rights)
        ),
        targets
    )
