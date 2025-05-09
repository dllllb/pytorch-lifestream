import numpy as np
from ptls.data_load.feature_dict import FeatureDict


class SeqLenLimit(FeatureDict):
    """
    This class is used as 'f_augmentation' argument for 
    ptls.data_load.datasets.augmentation_dataset.AugmentationDataset (AugmentationIterableDataset).

    Args:
        max_seq_len (int): maximum sequence length to keep
        strategy (str): strategy to use for truncating sequences. 
            Available options are 'tail', 'head' and 'random'. Default is 'tail'.

    """
    def __init__(self, 
                 max_seq_len: int, 
                 strategy: str = 'tail'):
        self.max_seq_len = max_seq_len
        self.strategy = strategy

        assert strategy in ('tail', 'head', 'random')

    def __call__(self, x: dict) -> dict:
        seq_len = self.get_seq_len(x)

        idx = self.get_idx(seq_len)
        new_x = self.seq_indexing(x, idx)
        return new_x

    def get_idx(self, seq_len: int) -> np.ndarray:
        ix = np.arange(seq_len)
        if self.strategy == 'tail':
            return slice(-self.max_seq_len, None)
        elif self.strategy == 'head':
            return slice(self.max_seq_len)
        elif self.strategy == 'random':
            if seq_len <= self.max_seq_len:
                return ix
            else:
                max_start_pos = seq_len - self.max_seq_len
                start_pos = np.random.choice(max_start_pos, 1)[0]
                return ix[start_pos:start_pos + self.max_seq_len]
