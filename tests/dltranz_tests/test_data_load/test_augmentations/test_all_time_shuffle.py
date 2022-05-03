import torch

from ptls.data_load.augmentations.all_time_shuffle import AllTimeShuffle


def test_shuffle():
    i_filter = AllTimeShuffle()
    data = {'event_time': torch.arange(5), 'mcc': torch.arange(5)}
    data = i_filter(data)
    assert len(data['event_time']) == 5
    assert data['mcc'].sum() == 10
