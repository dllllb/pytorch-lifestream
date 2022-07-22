import torch

from ptls.data_load.utils import collate_feature_dict


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
        if type(target_dtype) is str:
            self.target_dtype = getattr(torch, target_dtype)
        else:
            self.target_dtype = target_dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feature_arrays = self.data[item]
        return feature_arrays

    def __iter__(self):
        for feature_arrays in self.data:
            yield feature_arrays


    def collate_fn(self, padded_batch):
        padded_batch = collate_feature_dict(padded_batch)
        target = padded_batch.payload[self.target_col_name]
        del padded_batch.payload[self.target_col_name]
        if self.target_dtype is not None:
            target = target.to(dtype=self.target_dtype)
        return padded_batch, target


class SeqToTargetIterableDataset(SeqToTargetDataset, torch.utils.data.IterableDataset):
    pass
