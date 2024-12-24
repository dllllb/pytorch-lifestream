from functools import partial
import pytest
import os
import torch
import pandas as pd
import pytorch_lightning as pl
from sklearn.ensemble import RandomForestClassifier

from ptls.frames.inference_module import ONNXInferenceModule
from ptls.data_load.datasets.dataloaders import inference_data_loader
from ptls.preprocessing.pandas.pandas_preprocessor import PandasDataPreprocessor
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule


@pytest.fixture
def source_data():
    return pd.read_csv(
        'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz?download=true',
        compression='gzip')

@pytest.fixture
def preprocessor():
    return PandasDataPreprocessor(
        col_id='client_id',
        col_event_time='trans_date',
        event_time_transformation='none',
        cols_category=['small_group'],
        cols_numerical=['amount_rur'],
        return_records=True
    )

@pytest.fixture
def model():
    trx_encoder_params = dict(embeddings_noise=0.003,
                              numeric_values={'amount_rur': 'identity'},
                              embeddings={'trans_date': {'in': 800, 'out': 16},
                                          'small_group': {'in': 250, 'out': 16}})

    seq_encoder = RnnSeqEncoder(trx_encoder=TrxEncoder(**trx_encoder_params), hidden_size=256, type='gru')
    model = CoLESModule(seq_encoder=seq_encoder, optimizer_partial=partial(torch.optim.Adam, lr=0.001),
                        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9))
    return model


def test_onnx_export(preprocessor, source_data, model):
    dataset = preprocessor.fit_transform(source_data[:100])
    dl = inference_data_loader(
        dataset,
        batch_size=64,
        onnx=True
    )
    onnx_model_path = "test_model.onnx"
    onnx_module = ONNXInferenceModule(model=model, dl=dl, model_out_name=onnx_model_path)
    assert os.path.exists(onnx_model_path)
    batch = next(iter(dl))
    features, _, _ = onnx_module.preprocessing(batch)
    assert features.shape == onnx_module.model.example_input_array.shape
    os.remove(onnx_model_path)

def test_onnx_inference(preprocessor, source_data, model):
    dataset = preprocessor.fit_transform(source_data[:100])
    dl = inference_data_loader(
        dataset,
        batch_size=64,
        onnx=True
    )
    onnx_model_path = "test_model.onnx"
    exported_model = ONNXInferenceModule(model=model, dl=dl, model_out_name=onnx_model_path)
    batch = next(iter(dl))
    output_tensor = exported_model(batch)
    assert output_tensor.shape[1] > 0
    assert output_tensor.dtype == torch.float16
    os.remove(onnx_model_path)

def test_predict_with_onnx(preprocessor, source_data, model):
    dataset = preprocessor.fit_transform(source_data[:100])
    dl = inference_data_loader(
        dataset,
        batch_size=64,
        onnx=True
    )
    onnx_model_path = "test_model.onnx"
    onnx_model = ONNXInferenceModule(model=model, model_out_name=onnx_model_path, dl=dl)
    embeds = torch.vstack(pl.Trainer(accelerator="cpu", max_epochs=-1).predict(onnx_model, dl))
    assert embeds.shape[0] == len(dataset)
    assert embeds.shape[1] > 0
    df_target = pd.read_csv('https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/train_target.csv?download=true')
    df_target = df_target.set_index('client_id')
    df_target.rename(columns={"bins": "target"}, inplace=True)
    df = pd.DataFrame(data=embeds, columns=[f'embed_{i}' for i in range(embeds.shape[1])])
    df['client_id'] = [x['client_id'] for x in dataset]
    df = df.merge(df_target, how='left', on='client_id')
    embed_columns = [x for x in df.columns if x.startswith('embed')]
    x, y = df[embed_columns], df['target']
    clf = RandomForestClassifier()
    clf.fit(x, y)
    score = clf.score(x, y)
    assert score > 0.5
    os.remove(onnx_model_path)