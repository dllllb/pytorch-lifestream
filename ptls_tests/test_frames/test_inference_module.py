import hydra
import torch
from omegaconf import OmegaConf
import pandas as pd
from sklearn.metrics import roc_auc_score
from itertools import chain
import numpy as np

from ptls.data_load import PaddedBatch
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.inference_module import InferenceModule
import pytorch_lightning as pl

from ptls.nn import PBFeatureExtract
from ptls_tests.utils.data_generation import gen_trx_data


def get_rnn_seq_encoder():
    conf = OmegaConf.create("""
_target_: torch.nn.Sequential
_args_:
  - _target_: ptls.nn.RnnSeqEncoder
    trx_encoder:
      _target_: ptls.nn.TrxEncoder
      embeddings:
        mcc_code:
          in: 21
          out: 3
        trans_type:
          in: 11
          out: 2
      numeric_values:
        amount: log
    hidden_size: 16
  - _target_: ptls.nn.Head
    input_size: 16
    objective: classification
    num_classes: 1
    """)
    return hydra.utils.instantiate(conf)


def test_inference_module_predict():
    valid_loader = torch.utils.data.DataLoader(
        gen_trx_data((torch.rand(1000)*60+1).long(), target_type='bin_cls', use_feature_arrays_key=False),
        batch_size=64, collate_fn=collate_feature_dict)
    rnn_model = InferenceModule(
        model=get_rnn_seq_encoder(),
        model_out_name='pred',
    )

    df_out = pd.concat(pl.Trainer(accelerator="cpu", max_epochs=-1).predict(rnn_model, valid_loader))
    print(roc_auc_score(df_out['target'], df_out['pred']))


def test_score_model_mult2():
    valid_loader = [
        PaddedBatch(
            {'embeddings': torch.rand(4, 16), 'target_int': torch.arange(4), 'target_str': np.arange(4).astype(str)},
            torch.LongTensor([1, 3, 16, 5]),
        ),
        PaddedBatch(
            {'embeddings': torch.rand(2, 16), 'target_int': torch.arange(2), 'target_str': np.arange(2).astype(str)},
            torch.LongTensor([16, 2]),
        ),
        PaddedBatch(
            {'embeddings': torch.rand(1, 16), 'target_int': torch.arange(1), 'target_str': np.arange(1).astype(str)},
            torch.LongTensor([16]),
        ),
    ]
    model = InferenceModule(
        model=torch.nn.Sequential(
            PBFeatureExtract('embeddings', as_padded_batch=False),
            torch.nn.Linear(16, 2),
            torch.nn.Sigmoid(),
        ),
        model_out_name='pred', pandas_output=False,
    )

    dict_out = pl.Trainer(accelerator="cpu", max_epochs=-1).predict(model, iter(valid_loader))
    id1 = torch.cat([v['target_int'] for v in dict_out])
    id2 = np.concatenate([v['target_str'] for v in dict_out])

    assert torch.cat([v['pred'] for v in dict_out], dim=0).shape == (7, 2)
    assert id1.shape == (7,)
    assert id2.shape == (7,)

    np.testing.assert_array_almost_equal(id1, np.array([0, 1, 2, 3, 0, 1, 0]))
    assert id2.tolist() == ['0', '1', '2', '3', '0', '1', '0']

def get_rnn_seq_encoder_emb():
    conf = OmegaConf.create("""
_target_: ptls.nn.RnnSeqEncoder
trx_encoder:
  _target_: ptls.nn.TrxEncoder
  embeddings:
    mcc_code:
      in: 21
      out: 3
    trans_type:
      in: 11
      out: 2
  numeric_values:
    amount: log
hidden_size: 16
    """)
    return hydra.utils.instantiate(conf)


def test_inference_module_sequence():
    lengths = (torch.rand(1000)*60+1).long()
    trx_num = np.sum(lengths.numpy())
    trx_data = gen_trx_data(lengths, target_type='multi_cls', use_feature_arrays_key=False)
    valid_loader = torch.utils.data.DataLoader(
        trx_data,
        batch_size=64, collate_fn=collate_feature_dict)
    seq_enc = get_rnn_seq_encoder_emb()
    seq_enc.is_reduce_sequence = False
    rnn_model = InferenceModule(
        model=seq_enc,
        drop_seq_features=False,
        model_out_name='pred',
    )

    df_out = pd.concat(pl.Trainer(accelerator="cpu", max_epochs=-1).predict(rnn_model, valid_loader))
    assert df_out.shape == (trx_num, 20)
    assert list(df_out.mcc_code) == [mcc for usr in trx_data for mcc in usr['mcc_code']]

def test_inference_module_sequence_drop_seq():
    lengths = (torch.rand(1000)*60+1).long()
    trx_num = np.sum(lengths.numpy())
    trx_data = gen_trx_data(lengths, target_type='multi_cls', use_feature_arrays_key=False)
    valid_loader = torch.utils.data.DataLoader(
        trx_data,
        batch_size=64, collate_fn=collate_feature_dict)
    seq_enc = get_rnn_seq_encoder_emb()
    seq_enc.is_reduce_sequence = False
    rnn_model = InferenceModule(
        model=seq_enc,
        drop_seq_features=True,
        model_out_name='pred',
    )

    df_out = pd.concat(pl.Trainer(accelerator="cpu", max_epochs=-1).predict(rnn_model, valid_loader))
    assert df_out.shape == (trx_num, 17)
    assert list(df_out.target) == list(chain(*[[usr['target']]*usr['mcc_code'].shape[0] for usr in trx_data]))


def test_inference_module_record():
    lengths = (torch.rand(1000)*60+1).long()
    trx_data = gen_trx_data(lengths, target_type='multi_cls', use_feature_arrays_key=False)
    valid_loader = torch.utils.data.DataLoader(
        trx_data,
        batch_size=64, collate_fn=collate_feature_dict)
    seq_enc = get_rnn_seq_encoder_emb()
    seq_enc.is_reduce_sequence = True
    rnn_model = InferenceModule(
        model=seq_enc,
        drop_seq_features=False,
        model_out_name='pred',
    )

    df_out = pd.concat(pl.Trainer(accelerator="cpu", max_epochs=-1).predict(rnn_model, valid_loader))
    print(df_out)
    assert df_out.shape == (1000, 20)
    np.testing.assert_array_almost_equal(df_out.mcc_code.to_list()[0], trx_data[0]['mcc_code'].numpy())
    
def test_inference_module_record_drop_seq():
    lengths = (torch.rand(1000)*60+1).long()
    trx_data = gen_trx_data(lengths, target_type='multi_cls', use_feature_arrays_key=False)
    valid_loader = torch.utils.data.DataLoader(
        trx_data,
        batch_size=64, collate_fn=collate_feature_dict)
    seq_enc = get_rnn_seq_encoder_emb()
    seq_enc.is_reduce_sequence = True
    rnn_model = InferenceModule(
        model=seq_enc,
        drop_seq_features=True,
        model_out_name='pred',
    )

    df_out = pd.concat(pl.Trainer(accelerator="cpu", max_epochs=-1).predict(rnn_model, valid_loader))
    assert df_out.shape == (1000, 17)
    assert list(df_out.target)[0] == trx_data[0]['target']
