import torch
from ptls.data_load.feature_dict import FeatureDict


class AllTimeShuffle(FeatureDict):
    """Shuffle all transactions in event sequence
    """
    def __init__(self, event_time_name='event_time'):
        self.event_time_name = event_time_name

    @staticmethod
    def get_perm_ix(event_time):
        n = len(event_time)
        return torch.randperm(n)

    def __call__(self, x):
        ix = self.get_perm_ix(x[self.event_time_name])
        new_x = self.seq_indexing(x, ix)
        return new_x
