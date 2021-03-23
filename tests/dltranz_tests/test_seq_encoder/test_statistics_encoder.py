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
    neg = ('top_negative_trx', np.array([2010, 2370, 1010, 1110, 2330, 2371, 2011, 2020, 2331, 1100, 1030, 
                                         1200, 1210, 2210, 2021, 2110, 2340, 2440, 2460, 2320, 4010, 4071,
                                         2341, 2456, 4051, 1310, 1410, 4110, 2100, 2200, 4011, 1000, 4210,
                                         2446, 1510, 4020, 4500, 4041, 4090, 4031, 4021, 4097, 4100, 4061,
                                         2000, 4200, 4096, 4045, 4035]))
    pos = ('top_positive_trx', np.array([2010, 2370, 1010, 1110, 2330, 2371, 2011, 2020, 2331, 1100, 1030, 
                                         1200, 1210, 2210, 2021, 2110, 2340, 2440, 2460, 2320, 4010, 4071,
                                         2341, 2456, 4051, 1310, 1410, 4110, 2100, 2200, 4011, 1000, 4210,
                                         2446, 1510, 4020, 4500, 4041, 4090, 4031, 4021, 4097, 4100, 4061,
                                         2000, 4200, 4096, 4045, 4035]))
    model = StatisticsEncoder(dict([neg, pos]))

    x = get_data()

    out = model(x)
    assert isinstance(out, tuple) and len(out) == 4
    assert (abs(out[0] -  torch.Tensor([[-16.1181],
                                        [-16.1181],
                                        [-16.1181],
                                        [-16.1181]])) < torch.zeros((4, 1)) + eps).all()  
    assert out[1].shape == torch.Size([4, 6]) and out[1][0][3] == 0 and out[1][3][1] == 0
    assert out[2].shape == torch.Size([4, 1]) and abs(out[2][0].item() - 3.3029549820009882) < eps
    print(out[3].shape)
    print(abs(out[3][1][3].item())
    assert out[3].shape == torch.Size([4, 6]) and abs(out[3][1][3].item() - 0.7310606456724159) < eps
