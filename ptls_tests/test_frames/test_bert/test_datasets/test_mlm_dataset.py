import torch

from ptls.frames.bert.datasets.mlm_indexed_dataset import MlmIndexedDataset


def get_data():
    return [
        {
            'cat': torch.Tensor([1, 2, 3, 4, 5, 6]),
            'amnt': torch.Tensor([10, 12, 13, 41, 24]),
        },
        {
            'cat': torch.Tensor([9, 8, 7, 6, 5, 4, 3, 2]),
            'amnt': torch.Tensor([11, 16, 17, 16, 13, 10, 18, 16]),
        },
    ]


def test_mlm_dataset():
    ds = MlmIndexedDataset(
        data=get_data(),
        seq_len=3,
    )
    assert len(ds) == 5

    item0 = ds[0]
    assert item0['cat'].tolist() == [1, 2, 3]
    assert item0['amnt'].tolist() == [10, 12, 13]

    item4 = ds[4]
    assert item4['cat'].tolist() == [4, 3, 2]
    assert item4['amnt'].tolist() == [10, 18, 16]


def test_mlm_dataloader():
    ds = MlmIndexedDataset(
        data=get_data(),
        seq_len=3,
        step_rate=2/3,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=3, collate_fn=ds.collate_fn)
    assert len(dl) == 3

    for batch, exp_len in zip(dl, [3, 3, 1]):
        assert batch.payload['cat'].size(0) == exp_len
