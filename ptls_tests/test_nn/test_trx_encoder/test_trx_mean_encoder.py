from ptls.data_load import padded_collate, TrxDataset
from ptls.nn.trx_encoder.trx_mean_encoder import TrxMeanEncoder
from ptls_tests.utils.data_generation import gen_trx_data


def test_mean_encoder():
    x, y = padded_collate(TrxDataset(gen_trx_data([4, 3, 2])))

    params = {
        "embeddings_noise": .1,
        'embeddings': {
            'mcc_code': {'in': 21, 'out': 2},
            'trans_type': {'in': 11, 'out': 2},
        },
        'numeric_values': {'amount': 'log'}
    }

    te = TrxMeanEncoder(params)

    e = te(x)

    assert e.shape == (3, 33)
