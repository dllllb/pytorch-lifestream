import numpy as np
import torch
from ptls.data_load.utils import DictTransformer


def test_is_seq_feature_list():
    x = [1, 2, 3, 4]
    assert not DictTransformer.is_seq_feature(x)


def test_is_seq_feature_array():
    x = np.array([1, 2, 3, 4])
    assert DictTransformer.is_seq_feature(x)

def test_is_seq_feature_tensor():
    x = torch.Tensor([1, 2, 3, 4])
    assert DictTransformer.is_seq_feature(x)
