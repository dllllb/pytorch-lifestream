from torch.utils.data import Dataset, DataLoader

from ptls.metric_learn.dataset.infinite_loader import InfiniteBatchSampler


class TestInfiniteDataset(Dataset):
    def __getitem__(self, ix):
        return 1000 + ix


def test_sampler():
    data_loader = DataLoader(
        dataset=TestInfiniteDataset(),
        batch_sampler=InfiniteBatchSampler(epoch_size=5, batch_size=2),
        num_workers=0,
    )
    out_data = []
    for epoch in range(3):
        for batch in data_loader:
            out_data.append(batch.tolist())

    expected_out = [
        [1000, 1001],
        [1002, 1003],
        [1004],
        [1005, 1006],
        [1007, 1008],
        [1009],
        [1010, 1011],
        [1012, 1013],
        [1014],
    ]
    assert expected_out == out_data
