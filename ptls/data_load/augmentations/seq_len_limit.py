import numpy as np


class SeqLenLimit:
    def __init__(self, max_seq_len, strategy='tail'):
        self.max_seq_len = max_seq_len
        self.strategy = strategy

        assert strategy in ('tail', 'head', 'random')

    def __call__(self, x):
        seq_len = len(next(iter(x.values())))

        idx = self.get_idx(seq_len)
        new_x = {k: v[idx] for k, v in x.items()}
        return new_x

    def get_idx(self, seq_len):
        ix = np.arange(seq_len)
        if self.strategy == 'tail':
            return ix[-self.max_seq_len:]
        elif self.strategy == 'head':
            return ix[:self.max_seq_len]
        elif self.strategy == 'random':
            if seq_len <= self.max_seq_len:
                return ix
            else:
                max_start_pos = seq_len - self.max_seq_len
                start_pos = np.random.choice(max_start_pos, 1)[0]
                return ix[start_pos:start_pos + self.max_seq_len]
