import torch

from ptls.data_load import padded_collate_wo_target


class SeqToTargetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 target_col_name,
                 target_dtype=None,
                 *args, **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.data = data
        self.target_col_name = target_col_name
        self.target_dtype = target_dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feature_arrays = self.data[item]
        return self.get_target(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.get_target(feature_arrays)

    def get_target(self, feature_arrays):
        return {k: v for k, v in feature_arrays.items() if k != self.target_col_name}, \
               feature_arrays[self.target_col_name]

    def collate_fn(self, batch):
        features = [x for x, y in batch]
        target = [y for x, y in batch]
        return padded_collate_wo_target(features), torch.tensor(target, dtype=self.target_dtype)


class SeqToTargetIterableDataset(SeqToTargetDataset, torch.utils.data.IterableDataset):
    pass
