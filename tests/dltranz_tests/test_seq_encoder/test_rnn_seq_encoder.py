import torch
import numpy as np

from ptls.trx_encoder import PaddedBatch
from ptls.seq_encoder.rnn_encoder import RnnSeqEncoder


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
            'trainable_starter': 'static',
        }
    }

    model = RnnSeqEncoder(params, True)

    x = get_data()

    out = model(x)
    assert isinstance(out, torch.Tensor) and out.shape == torch.Size([4, 48])
