import torch
import pytorch_lightning as pl

from dltranz.seq_encoder import SequenceEncoder
from dltranz.models import Head
from dltranz.lightning_modules.emb_module import EmbModule
from dltranz.data_load.data_module.emb_data_module import EmbeddingDatasetFactory

from tests.dltranz_tests.test_data_load import RandomEventData, gen_trx_data


def test_train_inference():
    dm = RandomEventData()

    cat_sizes = {
        'mcc_code': 21,
        'trans_type': 11,
    }

    seq_encoder = SequenceEncoder(
        category_features=cat_sizes,
        numeric_features=[],
        trx_embedding_noize=0.003
    )

    head = Head(input_size=seq_encoder.embedding_size, use_norm_encoder=True)

    model = EmbModule(seq_encoder=seq_encoder, head=head)

    dlf = EmbeddingDatasetFactory(
        min_seq_len=25,
        seq_split_strategy='SampleSlices',
        category_names = model.seq_encoder.category_names,
        category_max_size = model.seq_encoder.category_max_size,
        split_count=5,
        split_cnt_min=25,
        split_cnt_max=200,
    )

    train = gen_trx_data((torch.rand(1000)*60+1).long())

    trainer = pl.Trainer(
        max_epochs=1,
        gpus=1 if torch.cuda.is_available() else 0
    )

    train_dl = dlf.train_data_loader(train, num_workers=1, batch_size=8)
    trainer.fit(model, train_dl)

    test = gen_trx_data((torch.rand(1000)*60+1).long())

    test_dl = dlf.inference_data_loader(test, num_workers=1, batch_size=8)
    trainer.predict(model, test_dl)
