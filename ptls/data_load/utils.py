import numpy as np
import torch

class DictTransformer:
    def __init__(self, *args, **kwargs):
        """Mixin constructor
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def is_seq_feature(x):
        """Check is value sequential feature

        Iterables are:
            np.array
            torch.Tensor

        Not iterable:
            list    - dont supports indexing

        Parameters
        ----------
        x:
            value for check

        Returns
        -------
            True if value is iterable
        """
        if type(x) in (np.ndarray, torch.Tensor):
            return True
        return False

    @staticmethod
    def seq_indexing(x, ix):
        """Apply indexing for seq_features only

        Parameters
        ----------
        x:
            value
        ix:
            indexes

        Returns
        -------

        """
        if DictTransformer.is_seq_feature(x):
            return x[ix]
        return x
