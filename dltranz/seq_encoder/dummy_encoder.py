import numpy as np
import torch

from experiments.scenario_gender.distribution_target import top_tr_types, get_distributions, load_data_pq, transform_inv
from dltranz.trx_encoder import PaddedBatch


class DummyEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dummy = torch.nn.Linear(1, 1)
        self.cat_names = ['mcc_code']
        self.num_values = []
        self.cat_max_size = {}
        self.embed_size = {}

    def forward(self, x):
        return x

    @property
    def category_names(self):
        return set(self.cat_names + self.num_values)

    @property
    def category_max_size(self):
        return self.cat_max_size

    @property
    def embedding_size(self):
        return self.embed_size
