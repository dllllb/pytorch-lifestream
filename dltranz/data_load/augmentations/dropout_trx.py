import numpy as np


class DropoutTrx:
    def __init__(self, trx_dropout):
        self.trx_dropout = trx_dropout

    def __call__(self, x):
        seq_len = len(next(iter(x.values())))

        idx = self.get_idx(seq_len)
        new_x = {k: v[idx] for k, v in x.items()}
        return new_x

    def get_idx(self, seq_len):
        if self.trx_dropout > 0 and seq_len > 0:
            idx = np.random.choice(seq_len, size=int(seq_len * (1 - self.trx_dropout)+1), replace=False)
            return np.sort(idx)
        else:
            return np.arange(seq_len)
