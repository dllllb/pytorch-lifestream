import pytorch_lightning as pl
import torch.optim
import torch


from pyhocon import ConfigFactory
from ptls.nn import Head, TrxEncoder
from functools import partial
from collections import defaultdict
from ptls.data_load.padded_batch import PaddedBatch

from ptls.frames.coles import CoLESModule
from ptls.frames.coles.multimodal_dataset import MultiModalIterableDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.frames.coles import MultiModalSortTimeSeqEncoderContainer
from ptls.nn.seq_encoder.rnn_encoder import RnnEncoder

def generate_multimodal_data(lengths, target_share=.5, target_type='bin_cls', use_feature_arrays_key=True):
    n = len(lengths)
    if target_type == 'bin_cls':
        targets = (torch.rand(n) >= target_share).long()
    else:
        raise AttributeError(f'Unknown target_type: {target_type}')
    sources = ['src1', 'src2']
    data_lst = []
    data = {}
    
    for target, length in zip(targets, lengths):
        for idx, src in enumerate(sources):
            data[f'src{idx+1}_event_time'] = (torch.rand(length)*100 + 1).long().sort().values
            data[f'src{idx+1}_trans_type_{idx+1}'] = (torch.rand(length)*10 + 1).long()
            data[f'src{idx+1}_mcc_code_{idx+1}'] = (torch.rand(length) * 20 + 1).long()
            data[f'src{idx+1}_amount_{idx+1}'] = (torch.rand(length) * 1000 + 1).long()
        data_lst.append(data)

    return data_lst

def create_train_mm_loader(data):
    
    dataset = MultiModalIterableDataset(
                                     data, 
                                     splitter = SampleSlices(split_count=5, cnt_min=20, cnt_max=200),
                                     col_id='epk_id',
                                     source_features={
                                         'src1': ['trans_type_1',
                                                  'mcc_code_1',
                                                  'amount_1',
                                                  'event_time',],
                                         'src2': ['trans_type_2',
                                                  'mcc_code_2',
                                                  'amount_2',
                                                  'event_time',],
                                     },
                                     source_names=['src1', 'src2']
                                    )
    
    dl = PtlsDataModule(
        train_data=dataset, train_num_workers=0, train_batch_size=4, 
        valid_data=dataset, valid_num_workers=0, valid_batch_size=1
    )
    return dl.train_dataloader()

class RandomMultimodalEventData(pl.LightningDataModule):
    def __init__(self, params, target_type='bin_cls'):
        super().__init__()
        self.hparams.update(params)
        self.target_type = target_type

    def train_dataloader(self):
        test_data = generate_multimodal_data((torch.rand(3) * 60 + 1).long(), target_type='bin_cls', use_feature_arrays_key=True)
        train_loader = create_train_mm_loader(test_data)
        return train_loader

    def test_dataloader(self):
        test_data = generate_multimodal_data((torch.rand(3) * 60 + 1).long(), target_type='bin_cls', use_feature_arrays_key=True)
        train_loader = create_train_mm_loader(test_data)
        return train_loader
    

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
        },
        "rnn": {
            "type": "gru",
            "input_size": 64,
            "seq_encoder_cls": RnnEncoder,
            "hidden_size": 16,
            "bidir": False,
            "trainable_starter": "static"
        },
        "trx_encoder_1": {
            "embeddings_noise": .003,
            "norm_embeddings": False,
            'embeddings': {
                'mcc_code_1': {'in': 21, 'out': 3},
                'trans_type_1': {'in': 11, 'out': 2},
            },
            'numeric_values': {'amount_1': 'log'},
            "linear_projection_size": 64
        },
        "trx_encoder_2": {
            "embeddings_noise": .003,
            "norm_embeddings": False,
            'embeddings': {
                'mcc_code_2': {'in': 21, 'out': 3},
                'trans_type_2': {'in': 11, 'out': 2},
            },
            'numeric_values': {'amount_2': 'log'},
            "linear_projection_size": 64
        },
    }

    params = ConfigFactory.from_dict(params)
    return params


def test_train_loop():
    params = tst_params()

    model = CoLESModule(
        seq_encoder=MultiModalSortTimeSeqEncoderContainer(
            trx_encoders={
                "src1": TrxEncoder(**params['trx_encoder_1']),
                "src2": TrxEncoder(**params['trx_encoder_2']),
            },
            **params['rnn'],
        ),
        head=Head(use_norm_encoder=True),
        optimizer_partial=partial(torch.optim.Adam),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=1.0),
    )
    dl = RandomMultimodalEventData(params['data_module'])
    trainer = pl.Trainer(
        max_epochs=1,
        logger=None,
        enable_checkpointing=False,
        accelerator='cpu'
    )
    trainer.fit(model, dl)
