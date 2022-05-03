import torch
from torch.utils.data import DataLoader

from ptls.data_load import padded_collate, ZeroDownSampler, DropoutTrxDataset, TrxDataset, LastKTrxDataset
from ptls.data_load import augmentation_chain
from ..test_trx_encoder import gen_trx_data


def test_padded_collate():
    data = [
        ({'a': torch.LongTensor([1, 2, 3, 4])}, torch.LongTensor([0])),
        ({'a': torch.LongTensor([1, 2])},  torch.LongTensor([0])),
        ({'a': torch.LongTensor([1])},  torch.LongTensor([1])),
    ]

    tt = torch.LongTensor([
        [1, 2, 3, 4],
        [1, 2, 0, 0],
        [1, 0, 0, 0]])

    x, y = padded_collate(data)

    assert x.payload['a'].shape == (3, 4)
    assert x.payload['a'].eq(tt).all()


def test_zero_down_sampler():
    y = torch.LongTensor([1, 0, 1, 0, 0, 0])
    sampler = ZeroDownSampler(y)
    idx = list(sampler)
    assert sum(y[idx]) == 2
    assert len(y[idx]) == 4


def test_data_loader():
    data = gen_trx_data((torch.rand(1000)*60+1).long())
    y0 = torch.LongTensor([e['target'] for e in data])
    ds = DropoutTrxDataset(TrxDataset(data), trx_dropout=0, seq_len=15)
    dl = DataLoader(ds, 10, collate_fn=padded_collate)
    y = torch.cat([y for x, y in dl])
    assert all(y0 == y)


def test_last_k_trx_dataset():
    data = gen_trx_data([100, 100, 100])
    res = [len(next(iter(x.values()))) for x, _ in LastKTrxDataset(TrxDataset(data), .5)]
    assert all(torch.tensor(res) == torch.tensor([50, 50, 50]))
    res = [len(next(iter(x.values()))) for x, _ in LastKTrxDataset(TrxDataset(data), .2)]
    assert all(torch.tensor(res) == torch.tensor([20, 20, 20]))


def _inc(x):
    return x + 1


def _double(x):
    return x * 2


def test_augmentation_chain():
    a = augmentation_chain(_inc, _double, _inc)
    out = a(2)
    assert out == 7


def test_augmentation_chain_pickle():
    import pickle

    a = augmentation_chain(_inc, _double)
    pickle.dumps(a)
