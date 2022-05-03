import torch
import numpy as np

from ptls.trx_encoder import PaddedBatch
from ptls.seq_encoder.rnn_encoder import RnnSeqEncoderDistributionTarget


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
    eps = 1e-5

    params = {
        'trx_encoder' : {
            'norm_embeddings': False,
            'embeddings_noise': 0.003,
            'embeddings': {
                'mcc_code': {
                    'in': 200,
                    'out': 48
                },
                'tr_type': {
                    'in': 100,
                    'out': 24
                }
            },
            'numeric_values': {
                'amount': 'identity'
            },
        },
        'rnn': {
            'hidden_size': 48,
            'type': 'gru',
            'bidir': False,
            'trainable_starter': 'static'
        },
        'head_layers': {
            'CombinedTargetHeadFromRnn': {
                'in_size': 48,
                'num_distr_classes': 6,
                'pos': True,
                'neg': True,
                'use_gates': True,
                'pass_samples': True
            }
        }
    }

    model = RnnSeqEncoderDistributionTarget(params, True)

    x = get_data()

    out = model(x)
    assert isinstance(out, tuple) and len(out) == 3
    assert isinstance(out[0], torch.Tensor) and out[0].shape == torch.Size([4, 48])
    assert (out[1] - np.array([-16.118095, -16.118095, -16.118095, -16.118095]) < np.zeros((1, 4)) + eps).all()
    assert (out[2] - np.array([3.302955, 11.313237, 25.456194, 37.45834])< np.zeros((1, 4)) + eps).all()
