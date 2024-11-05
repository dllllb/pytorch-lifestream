import numpy as np
import torch


class FeatureDict:
    """Tools for feature-dict format

    Args:
        Feature dict:
            keys are feature names
            values are feature values, sequential, scalar or arrays

    """
    def __init__(self, *args, **kwargs):
        """Mixin constructor
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def is_seq_feature(k: str=None, x=None) -> bool:
        """Check is value sequential feature
        Synchronized with ptls.data_load.padded_batch.PaddedBatch.is_seq_feature

        Args:
            k: feature_name - `np.array` or `torch.Tensor`. `list` is Not iterable:
                - dont supports indexing
            x: value for check

        Returns:
            True if value is iterable
        
        """
        
        if k == 'event_time':
            return True
        
        if k.startswith('target'):
            return False

        if isinstance(x, (np.ndarray, torch.Tensor)):
            return True
        return False

    @staticmethod
    def seq_indexing(d: dict, ix: int) -> dict:
        """Apply indexing for seq_features only

        Args:
            d: feature dict
            ix: indexes
        
        Returns:
            dict

        """
        return {k: v[ix] if FeatureDict.is_seq_feature(k, v) else v for k, v in d.items()}

    @staticmethod
    def get_seq_len(d) -> int:
        """Finds a sequence column and return its length

        Args:
            d: feature-dict
        
        Returns:
            int

        """
        if 'event_time' in d:
            return len(d['event_time'])
        return len(next(v for k, v in d.items() if FeatureDict.is_seq_feature(k, v)))
