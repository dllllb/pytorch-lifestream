import torch

from ptls.seq_encoder import SequenceEncoder
from ptls.trx_encoder import PaddedBatch


def get_data():
    payload = {'amount': torch.arange(4 * 10).view(4, 10).float(),
               'event_time': torch.arange(4 * 10).view(4, 10).float(),
               'mcc_code': torch.arange(4 * 10).view(4, 10),
               'tr_type': torch.arange(4 * 10).view(4, 10)
               }
    return PaddedBatch(
        payload=payload,
        length=torch.tensor([4, 2, 6, 8])
    )


def test_shape():
    model = SequenceEncoder(
        category_features={'mcc_code': 200, 'tr_type': 100},
        numeric_features=["amount"],
        rnn_hidden_size=48
    )

    x = get_data()

    out = model(x)
    assert isinstance(out, torch.Tensor) and out.shape == torch.Size([4, 48])
