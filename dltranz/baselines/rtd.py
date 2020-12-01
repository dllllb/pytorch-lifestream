import torch

from dltranz.data_load import padded_collate_wo_target
from dltranz.trx_encoder import PaddedBatch


def collate_rtd_batch(batch, replace_prob, skip_first=0):
    padded_batch = padded_collate_wo_target(batch)

    new_x, lengths = padded_batch.payload, padded_batch.seq_lens

    mask = torch.zeros(next(iter(new_x.values())).shape, dtype=torch.int64)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1

    to_replace = torch.bernoulli(mask * replace_prob).bool()
    to_replace[:, :skip_first] = False

    sampled_trx_ids = torch.multinomial(
        mask.flatten().float(),
        num_samples=to_replace.sum().item(),
        replacement=True
    )

    to_replace_flatten = to_replace.flatten()
    for k, v in new_x.items():
        v.flatten()[to_replace_flatten] = v.flatten()[sampled_trx_ids]

    return PaddedBatch(new_x, lengths), to_replace.long().flatten()[mask.flatten().bool()]
