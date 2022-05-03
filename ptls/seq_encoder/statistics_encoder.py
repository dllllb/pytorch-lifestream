import numpy as np
import torch

from ptls.seq_encoder.utils import top_tr_types, get_distributions, load_data_pq, transform_inv
from ptls.trx_encoder import PaddedBatch


class StatisticsEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.collect_pos, self.collect_neg = config.get('pos', True), config.get('neg', True)
        self.dummy = torch.nn.Linear(1, 1)
        self.negative_items = config.get('top_negative_trx', [])
        self.positive_items = config.get('top_positive_trx', [])

        self.cat_names = list(config.get('category_names'))
        self.num_values = list(config.get('numeric_values'))
        self.cat_max_size = config.get('category_max_size')

    def forward(self, x: PaddedBatch):
        eps = 1e-7
        tr_type_col = []
        amount_col = []
        for i, row in enumerate(zip(x.payload[self.cat_names[0]], x.payload[self.num_values[0]])):
            tr_type_col += [list(row[0][:x.seq_lens[i].item()].cpu().numpy())]
            amount_col += [list(row[1][:x.seq_lens[i].item()].cpu().numpy())]

        tr_type_col = np.array(tr_type_col, dtype=object)[:, None]
        amount_col = np.array(amount_col, dtype=object)[:, None]
        np_data = np.hstack((tr_type_col, amount_col))

        distr = get_distributions(np_data, 1, 0, self.negative_items, self.positive_items, 5, 0, transform_inv)

        if self.collect_neg:
            sums_of_negative_target = torch.tensor(distr[0])[:, None]
            neg_distribution = torch.tensor(distr[2])
            log_neg_sum = torch.log(torch.abs(sums_of_negative_target + eps))
        if self.collect_pos:
            sums_of_positive_target = torch.tensor(distr[1])[:, None]
            pos_distribution = torch.tensor(distr[3])
            log_pos_sum = torch.log(sums_of_positive_target + eps)

        if self.collect_neg and self.collect_pos:
            return log_neg_sum, neg_distribution, log_pos_sum, pos_distribution
        elif self.collect_neg:
            return log_neg_sum, neg_distribution, 0, 0
        elif self.collect_pos:
            return 0, 0, log_pos_sum, pos_distribution

    @property
    def category_names(self):
        return set(self.cat_names + self.num_values)

    @property
    def category_max_size(self):
        return self.cat_max_size
