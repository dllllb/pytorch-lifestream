from typing import Dict

import numpy as np
import torch

from ptls.data_load.feature_dict import FeatureDict


class PaddedBatch:
    """Contains a padded batch of sequences with different lengths.

    Parameters:
        payload:
            container with data. Format supported:
            - dict with features. This is the input data for overall network pipeline.
                Kees are the feature names, values are (B, T) shape tensors.
                Long type for categorical features, embedding lookup table indexes expected
                Float type for numerical features.
            - trx embedding tensor. This is the intermediate data for overall network pipeline.
                shape (B, T, H)
            - feature tensor. Used in some cases
                shape (B, T)
        length:
            Tensor of shape (B,) with lengths of sequences.
            All sequences in `payload` has length T, but only L first are used.
            Unused positions padded with zeros

    Examples:
        >>> # Dict with features
        >>>
        >>> data = PaddedBatch(
        >>>     payload={
        >>>         'mcc': torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]])
        >>>         'amnt': torch.tensor([[90, 50, 0, 0], [40, 10, 55, 70]])
        >>>     },
        >>>     length=torch.Tensor([2, 4])  
        >>> )
        >>>
        >>> # check shape
        >>> >> torch.testing.assert_close(data.payload['mcc'].size(), (2, 4))
        >>> 
        >>> # check first transaction
        >>> torch.testing.assert_close(data.payload['mcc'][:, 0], torch.tensor([1, 3]))
        >>>
        >>> # get last transaction
        >>> torch.testing.assert_close(data.payload['mcc'][torch.arange(2), data.seq_lens - 1], torch.tensor([2, 6]))
        >>> 
        >>> # get all transaction flatten
        >>> torch.testing.assert_close(data.payload['mcc'][data.seq_len_mask.bool()], torch.tensor([1, 2, 3, 4, 5, 6]))
        >>>
        >>> # Feature tensor
        >>>
        >>> data = PaddedBatch(
        >>>     payload=torch.tensor([
        >>>         [1, 2, 0, 0],
        >>>         [3, 4, 5, 6],
        >>>         [7, 8, 9, 0],
        >>>     ]),
        >>>     length=torch.tensor([2, 4, 3]),
        >>> )
        >>>
        >>> # check shape
        >>> torch.testing.assert_close(data.payload.size(), (3, 4))
        >>>
        >>> # get first transaction
        >>> torch.testing.assert_close(data.payload[:, 0], torch.tensor([1, 3, 7]))
        >>>
        >>> # get last transaction
        >>> torch.testing.assert_close(data.payload[torch.arange(3), data.seq_lens - 1], torch.tensor([2, 6, 9]))
        >>>
        >>> # get all transaction flatten
        >>> torch.testing.assert_close(data.payload[data.seq_len_mask.bool()], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    """
    def __init__(self, payload: Dict[str, torch.Tensor], length: torch.LongTensor):
        self._payload = payload
        self._length = length

    @property
    def payload(self):
        return self._payload

    @property
    def seq_lens(self):
        return self._length

    @property
    def device(self):
        return self._length.device

    @property
    def seq_feature_shape(self):
        return next(v.size() for k, v in self._payload.items() if self.is_seq_feature(k, v))

    def __len__(self):
        return len(self._length)

    def to(self, device, non_blocking=False):
        length = self._length.to(device=device, non_blocking=non_blocking)
        if type(self._payload) == dict:
            payload = {
                k: v.to(device=device, non_blocking=non_blocking) if type(v) is torch.Tensor else v
                for k, v in self._payload.items()
            }
        elif type(self._payload) == torch.Tensor:
            payload = self._payload.to(device=device, non_blocking=non_blocking)
        return PaddedBatch(payload, length)

    @property
    def seq_len_mask(self):
        """mask with B*T size for valid tokens in `payload`
        """
        if type(self._payload) is dict:
            B, T = next(v for k, v in self._payload.items() if self.is_seq_feature(k, v)).size()
        else:
            B, T = self._payload.size()[:2]
        return (torch.arange(T, device=self._length.device).unsqueeze(0).expand(B, T) <                 self._length.unsqueeze(1)).long()

    @staticmethod
    def is_seq_feature(k: str, x):
        """Check is value sequential feature
        Synchronized with ptls.data_load.feature_dict.FeatureDict.is_seq_feature

                     1-d        2-d
        event_time | True      True
        target_    | False     False  # from FeatureDict.is_seq_feature
        tensor     | False     True

        Parameters
        ----------
        k:
            feature_name
        x:
            value for check

        Returns
        -------

        """
        if k == 'event_time':
            return True
        if k.startswith('target'):
            return False
        if type(x) is np.ndarray:
                return False
        if type(x) is torch.Tensor and len(x.shape) == 1:
            return False
        return True 

    def drop_seq_features(self):
        """Returns new dict without sequential features

        Returns
        -------

        """
        return {k: v for k, v in self.payload.items() if not PaddedBatch.is_seq_feature(k, v)}

    def keep_seq_features(self):
        """Returns new PaddedBatch with sequential features only

        Returns
        -------

        """
        return PaddedBatch(
            payload={k: v for k, v in self.payload.items() if PaddedBatch.is_seq_feature(k, v)},
            length=self.seq_lens,
        )
