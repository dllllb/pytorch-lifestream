import torch

from ptls.data_load import PaddedBatch


class PBFeatureExtract(torch.nn.Module):
    def __init__(self, feature_col_name, as_padded_batch):
        super().__init__()

        self.feature_col_name = feature_col_name
        self.as_padded_batch = as_padded_batch

    def forward(self, x: PaddedBatch):
        feature = x.payload[self.feature_col_name]
        if self.as_padded_batch:
            return PaddedBatch(feature, x.seq_lens)
        return feature
