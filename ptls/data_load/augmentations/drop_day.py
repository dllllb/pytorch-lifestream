import numpy as np
import torch
from ptls.data_load.utils import DictTransformer


class DropDay(DictTransformer):
    def __init__(self, event_time_name='event_time'):
        self.event_time_name = event_time_name

    def __call__(self, x):
        mask = self.get_perm_ix(x[self.event_time_name])
        new_x = {k: self.seq_indexing(k, v, mask) for k, v in x.items()}
        return new_x

    @staticmethod
    def get_perm_ix(event_time):
        days = torch.unique(event_time, sorted=True)
        ix = np.random.choice(len(days), 1)[0]
        mask = event_time != days[ix]
        return mask
