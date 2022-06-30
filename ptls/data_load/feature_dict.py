import numpy as np
import torch


class FeatureDict:
    """Tools for feature-dict format
    Feature dict:
        keys are feature names
        values are feature values, sequential, scalar or arrays
    """
    def __init__(self, *args, **kwargs):
        """Mixin constructor
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def is_seq_feature(k: str, x):
        """Check is value sequential feature
        Synchronized with ptls.data_load.padded_batch.PaddedBatch.is_seq_feature

        Iterables are:
            np.array
            torch.Tensor

        Not iterable:
            list    - dont supports indexing

        Parameters
        ----------
        k:
            feature_name
        x:
            value for check

        Returns
        -------
            True if value is iterable
        """
        if k == 'event_time':
            return True
        if k.startswith('target'):
            return False
        if type(x) in (np.ndarray, torch.Tensor):
            return True
        return False

    @staticmethod
    def seq_indexing(d, ix):
        """Apply indexing for seq_features only

        Parameters
        ----------
        d:
            feature dict
        ix:
            indexes

        Returns
        -------

        """
        return {k: v[ix] if FeatureDict.is_seq_feature(k, v) else v for k, v in d.items()}

    @staticmethod
    def get_seq_len(d):
        """Finds a sequence column and return its length

        Parameters
        ----------
        d:
            feature-dict

        """
        if 'event_time' in d:
            return len(d['event_time'])
        return len(next(v for k, v in d.items() if FeatureDict.is_seq_feature(k, v)))
