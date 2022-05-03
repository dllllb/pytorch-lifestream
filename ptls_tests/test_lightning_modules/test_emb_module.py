import pytorch_lightning as pl

from pyhocon import ConfigFactory

from ptls.lightning_modules.emb_module import EmbModule
from ptls.models import Head
from ptls.seq_encoder import SequenceEncoder
from ..test_data_load import RandomEventData


def tst_params():
    params = {
        "data_module": {
            "train": {
                "num_workers": 1,
                "batch_size": 32,
                "trx_dropout": 0.01,
                "max_seq_len": 100,
            },
            "valid": {
                "batch_size": 16,
                "num_workers": 1,
                "max_seq_len": 100
            }
        }
    }

    params = ConfigFactory.from_dict(params)
    return params


def test_train_loop():
    seq_encoder = SequenceEncoder(
        category_features={'mcc_code': 21, 'trans_type': 11},
        numeric_features=["amount"],
    )
    head = Head(
        input_size=seq_encoder.embedding_size,
        use_norm_encoder=True
    )
    model = EmbModule(
        seq_encoder=seq_encoder,
        head=head,
    )

    params = tst_params()

    dl = RandomEventData(params['data_module'])
    trainer = pl.Trainer(max_epochs=1, logger=None, checkpoint_callback=False)
    trainer.fit(model, dl)
