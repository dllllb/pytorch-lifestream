import pandas as pd
import torch
from joblib import cpu_count

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from sklearn.model_selection import train_test_split
from functools import partial
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule
from ptls.preprocessing.dask.dask_preprocessor import DaskDataPreprocessor
from ptls.preprocessing.pandas.pandas_preprocessor import PandasDataPreprocessor
from pyinstrument import Profiler
import pytorch_lightning as pl

profiler = Profiler()


def define_data():
    source_data = pd.read_csv(
        'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz?download=true',
        compression='gzip')

    preprocessor = PandasDataPreprocessor(
        col_id='client_id',
        col_event_time='trans_date',
        event_time_transformation='none',
        cols_category=['small_group'],
        cols_numerical=['amount_rur'],
        return_records=True
    )
    params = dict(col_id='client_id',
                  col_event_time='trans_date',
                  event_time_transformation='none',
                  cols_category=['small_group'],
                  cols_numerical=['amount_rur'],
                  return_records=True)
    preprocessor = DaskDataPreprocessor(params)
    return preprocessor, source_data


def define_model():
    trx_encoder_params = dict(embeddings_noise=0.003,
                              numeric_values={'amount_rur': 'identity'},
                              embeddings={'trans_date': {'in': 800, 'out': 16},
                                          'small_group': {'in': 250, 'out': 16}})

    seq_encoder = RnnSeqEncoder(trx_encoder=TrxEncoder(**trx_encoder_params), hidden_size=256, type='gru')
    model = CoLESModule(seq_encoder=seq_encoder, optimizer_partial=partial(torch.optim.Adam, lr=0.001),
                        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9))
    return model


if __name__ == "__main__":
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    device = 1 if torch.cuda.is_available() else "auto"
    data_loader_workers = 1 if torch.cuda.is_available() else cpu_count()
    preprocessor, source_data = define_data()
    model = define_model()

    profiler.start()
    dataset = preprocessor.fit_transform(source_data)
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    len_filter = SeqLenFilter(min_seq_len=25)
    in_memory_dataset = MemoryMapDataset(data=train, i_filters=[len_filter])
    data_splitter = SampleSlices(split_count=5, cnt_min=25, cnt_max=200)
    coles_df = ColesDataset(data=in_memory_dataset, splitter=data_splitter)
    train_dl = PtlsDataModule(
        train_data=coles_df,
        train_num_workers=data_loader_workers,
        train_batch_size=256,
    )
    trainer = pl.Trainer(
        max_epochs=15,
        accelerator=accelerator,
        devices=device,
        enable_progress_bar=True,
        # prepare_data_per_node=False,
        # replace_sampler_ddp=False,
        # sync_batchnorm=True
    )
    print(f'logger.version = {trainer.logger.version}')
    trainer.fit(model, train_dl)
    print(trainer.logged_metrics)
    profiler.stop()
    profiler.print()
