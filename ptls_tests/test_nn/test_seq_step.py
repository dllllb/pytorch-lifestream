import torch

from ptls.nn import TimeStepShuffle
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_step import SkipStepEncoder, LastStepEncoder


def test_timestep_shuffle():
    t = torch.tensor([
        [[0, 0], [1, 2], [3, 4], [0, 0]],
        [[0, 0], [10, 11], [0, 0], [0, 0]],
    ])

    res = TimeStepShuffle()(PaddedBatch(t, [2, 1]))

    assert res.payload.shape == (2, 4, 2)


def test_skip_step_encoder():
    t = torch.arange(8*11*2).view(8, 11, 2)

    res = SkipStepEncoder(3)(PaddedBatch(t, [10, 9, 8, 7, 3, 2, 1, 0]))

    assert res.payload.shape == (8, 4, 2)

def test_last_step_encoder_padding():
    t = torch.tensor([
        [[1, 1], [2, 2], [3, 3], [0, 0], [0, 0]],  
        [[4, 4], [5, 5], [0, 0], [0, 0], [0, 0]],  
        [[6, 6], [0, 0], [0, 0], [0, 0], [0, 0]],  
    ])
    
    seq_lens = torch.tensor([3, 2, 1])
    encoder = LastStepEncoder()
    
    
    result = encoder(PaddedBatch(t, seq_lens))
    
    
    expected = torch.tensor([
        [3, 3],  
        [5, 5],  
        [6, 6],  
    ])

    assert torch.allclose(result, expected), "LastStepEncoder wrong indexation"