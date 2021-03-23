import numpy as np
import torch

from experiments.scenario_gender.distribution_target import top_tr_types, get_distributions, load_data_pq, transform_inv
from dltranz.trx_encoder import PaddedBatch


class StatisticsEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dummy = torch.nn.Linear(1, 1)
        self.negative_items = config['top_negative_trx']
        self.positive_items = config['top_positive_trx']

    def forward(self, x: PaddedBatch):
        eps = 1e-7
        tr_type_col = []
        amount_col = []
        for i, row in enumerate(zip(x.payload['tr_type'], x.payload['amount'])):
            tr_type_col += [list(row[0][:x.seq_lens[i].item()].cpu().numpy())]
            amount_col += [list(row[1][:x.seq_lens[i].item()].cpu().numpy())]

        tr_type_col = np.array(tr_type_col, dtype=object)[:, None]
        amount_col = np.array(amount_col, dtype=object)[:, None]
        np_data = np.hstack((tr_type_col, amount_col))

        distr = get_distributions(np_data, 1, 0, self.negative_items, self.positive_items, 5, 0, transform_inv)
        sums_of_negative_target, sums_of_positive_target = torch.tensor(distr[0])[:, None], torch.tensor(distr[1])[:, None]
        log_neg_sum = torch.log(torch.abs(sums_of_negative_target + eps))
        log_pos_sum = torch.log(sums_of_positive_target + eps)
        neg_distribution, pos_distribution = torch.tensor(distr[2]), torch.tensor(distr[3])
        
        return log_neg_sum, neg_distribution, log_pos_sum, pos_distribution

    @property
    def category_names(self):
        return set(['tr_type', 'mcc_code', 'amount'])

    @property
    def category_max_size(self):
        return {'mcc_code': 200, 'tr_type': 100}
