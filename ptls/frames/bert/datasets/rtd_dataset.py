import torch

from ptls.data_load import padded_collate_wo_target
from ptls.data_load.augmentations.random_slice import RandomSlice
from ptls.nn import PaddedBatch
from copy import deepcopy


class RtdDataset(torch.utils.data.Dataset):
    def __init__(self, data,
                 min_len, max_len, rate_for_min: float = 1.0,
                 replace_prob=0.15,
                 skip_first=1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.data = data
        self.r_slice = RandomSlice(min_len, max_len, rate_for_min)

        self.replace_prob = replace_prob
        self.skip_first = skip_first

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feature_arrays = self.data[item]
        return self.r_slice(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.r_slice(feature_arrays)

    def collate_fn(self, batch):
        padded_batch = padded_collate_wo_target(batch)

        new_x, lengths, mask = padded_batch.payload, padded_batch.seq_lens, padded_batch.seq_len_mask

        to_replace = torch.bernoulli(mask * self.replace_prob).bool()
        to_replace[:, :self.skip_first] = False

        sampled_trx_ids = torch.multinomial(
            mask.flatten().float(),
            num_samples=to_replace.sum().item(),
            replacement=True,
        )

        to_replace_flatten = to_replace.flatten()
        new_x = deepcopy(new_x)
        for k, v in new_x.items():
            v.flatten()[to_replace_flatten] = v.flatten()[sampled_trx_ids]

        return PaddedBatch(new_x, lengths), to_replace.long().float().flatten()[mask.flatten().bool()]


class RtdIterableDataset(RtdDataset, torch.utils.data.IterableDataset):
    pass
