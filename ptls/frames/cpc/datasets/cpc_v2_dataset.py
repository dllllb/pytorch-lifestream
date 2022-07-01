from ptls.frames.coles.coles_dataset import ColesDataset
from collections import defaultdict
import torch
from ptls.data_load.padded_batch import PaddedBatch


class CpcV2Dataset(ColesDataset):
    @staticmethod
    def collate_fn(batch):
        # TODO: refactoring required

        split_count = len(batch[0])

        sequences = [defaultdict(list) for _ in range(split_count)]
        lengths = torch.zeros((len(batch), split_count), dtype=torch.int)

        for i, client_data in enumerate(batch):
            for j, subseq in enumerate(client_data):
                for k, v in subseq.items():
                    sequences[j][k].append(v)
                    lengths[i][j] = v.shape[0]

        # adding padds in each split
        padded_batch = []
        for j, seq in enumerate(sequences):
            for k, v in seq.items():
                seq[k] = torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
            padded_batch.append(PaddedBatch(seq, lengths[:, j].long()))

        return tuple(padded_batch)


class CpcV2IterableDataset(CpcV2Dataset, torch.utils.data.IterableDataset):
    pass
