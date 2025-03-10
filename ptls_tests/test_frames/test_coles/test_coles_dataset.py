import numpy as np
import torch

from ptls.frames.coles import ColesDataset
from ptls.data_load import TrxDataset
from ptls.frames.coles.split_strategy import SampleSlices


def gen_trx_data(lengths):
    n = len(lengths)
    targets = (torch.rand(n) >= 0.5).long()
    samples = list()
    for target, length in zip(targets, lengths):
        s = dict()
        s['trans_type'] = (torch.rand(length)*10 + 1).long()
        s['event_time'] = (torch.rand(length)*10 + 1).long()
        s['mcc_code'] = (torch.rand(length) * 20 + 1).long()
        s['amount'] = (torch.rand(length) * 1000 + 1).long()
        s['target'] = target
        s['str_feature'] = "some_str_feature"
        
        samples.append({'feature_arrays': s})
    return samples


def test_train_loop():    
    test_data = TrxDataset(
        gen_trx_data((torch.rand(1000) * 60 + 1).long()),
        with_target = False,
    )
    splitter = SampleSlices(
        split_count = 3,
        cnt_min = 10,
        cnt_max = 60,
    )
    coles_data = ColesDataset(test_data, splitter=splitter)

    _ = next(iter(coles_data)) # No Error
