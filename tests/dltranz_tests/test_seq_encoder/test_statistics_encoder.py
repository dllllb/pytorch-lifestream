import torch
import numpy as np

from dltranz.trx_encoder import PaddedBatch
from dltranz.seq_encoder.statistics_encoder import StatisticsEncoder


def get_data():
    payload = {'amount': torch.arange(4*10).view(4, 10).float(),
               'event_time': torch.arange(4*10).view(4, 10).float(),
               'mcc_code': torch.arange(4*10).view(4, 10),
               'tr_type': torch.arange(4*10).view(4, 10)
              }
    return PaddedBatch(
                       payload=payload,
                       length=torch.tensor([4, 2, 6, 8])
                      )

def test_shape():
    eps = 1e-4

    model = StatisticsEncoder({})

    x = get_data()

    out = model(x)
    assert isinstance(out, tuple) and len(out) == 4
    print('111111111111111111')
    assert (abs(out[0] -  torch.Tensor([[-16.1181],
                                        [-16.1181],
                                        [-16.1181],
                                        [-16.1181]])) < torch.zeros((4, 1)) + eps).all()
    print('222222222222222222')      
    assert out[1].shape == torch.Size([4, 6]) and out[1][0][3] == 0 and out[1][3][1] == 0
    print('333333333333333333')
    assert out[2].shape == torch.Size([4, 1]) and abs(out[2][0].item() - 3.3029549820009882) < eps
    print('444444444444444444')
    assert out[3].shape == torch.Size([4, 6]) and abs(out[3][1][3].item() - 0.7310606456724159) < eps
