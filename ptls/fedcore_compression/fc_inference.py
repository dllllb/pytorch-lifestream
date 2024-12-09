"""Module for embeddings obtaining or/and computational metrics evaluation."""
import logging
import os

from pathlib import Path
from typing import Any, Literal, Union

import hydra
from omegaconf import DictConfig
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ptls.data_load.utils import collate_feature_dict
from ptls.fedcore_compression.fc_utils import eval_computational_metrics
from ptls.frames.inference_module import InferenceModule

logger = logging.getLogger(__name__)


def collate_feature_dict_for_perf_eval(batch: Any):
    """Auxiliary function to wrap batches"""
    return collate_feature_dict(batch), None


def save_scores(df_scores: pd.DataFrame, 
                output_path: Union[str, Path], 
                additional: str = '', 
                output_format: Literal['csv', 'parquet', 'pickle'] = 'csv'):
    """
    Saves embeddings to specified location.

    Parameters:
        df_scores: obtained_embeddings
        output_path: file destination
        additional: auxiliary identifier to differentiate obtained results
        output_format: how to save 
    """
    if output_format not in ('pickle', 'csv', 'parquet'):
        logger.warning(f'Format "{output_format}" is not supported. Used default "pickle"')
        output_format = 'pickle'

    output_path = Path(output_path, f'scores{additional}.{output_format}')

    if output_format == 'pickle':
        df_scores.to_pickle(output_path)
    elif output_format == 'csv':
        df_scores.to_csv(output_path, sep=',', header=True, index=False)
    elif output_format == 'parquet':
        df_scores.to_parquet(output_path)
    else:
        raise AssertionError('Never happens')
    logger.info(f'{len(df_scores)} records saved to: "{output_path}"')


def inference_run(path: Union[str, Path], conf: DictConfig, id: str = ''):
    """
    Runs inference for a pretrained encoder. 
    Also, if ``n_batches_computational`` specified in `conf` the computational metrics are calculated
    and saved into `computational.txt`

    Parameters:
        path: path to pretrained encoder
        conf: config specifying `.inference.dataset` 
        id: auxiliary identifier to differentiate obtained results
    """
    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(path)
    seq_encoder = model.seq_encoder if hasattr(model, 'seq_encoder') else model
    model = InferenceModule(
        model=seq_encoder,
        pandas_output=True, model_out_name='emb',
    )
    model.model.is_reduce_sequence = True

    dataset_inference = hydra.utils.instantiate(conf.inference.dataset)
    inference_dl = DataLoader(
        dataset=dataset_inference,
        collate_fn=collate_feature_dict,
        shuffle=False,
        num_workers=conf.inference.get('num_workers', 0),
        batch_size=conf.inference.get('batch_size', 128),
    )

    if conf.get('n_batches_computational', 0):
        eval_computational_metrics(
            model, 
            DataLoader(
                dataset=dataset_inference,
                collate_fn=collate_feature_dict_for_perf_eval,
                shuffle=False,
                num_workers=conf.inference.get('num_workers', 0),
                batch_size=conf.inference.get('batch_size', 32),
            ),
            save_path=(path.parent / 'computational.txt' 
                       if not path.is_dir() 
                       else Path(path, 'computational.txt')), 
            id=id,
            n_batches=conf.n_batches_computational
        )
    df_scores = pl.Trainer(accelerator=accelerator, 
                            limit_predict_batches=conf.get('limit_predict_batches', None),
                            limit_test_batches=conf.get('limit_test_batches', None),
                            limit_val_batches=conf.get('limit_val_batches', None),
                            limit_train_batches=conf.get('limit_train_batches', None),
                            max_epochs=-1).predict(model, inference_dl)
    df_scores = pd.concat(df_scores, axis=0)
    logger.info(f'df_scores examples: {df_scores.shape}:') 
    save_scores(df_scores, conf.inference.output, additional=id, output_format='csv')

@hydra.main(version_base='1.2', config_path=None)
@torch.no_grad()
def main(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)
    path = Path(conf.inference_model)
    if path.is_dir():
        for i, ckpt_name in tqdm(enumerate(os.listdir(path)), 'Checkpoint #'):
            if not (ckpt_name.endswith('.pth') or ckpt_name.endswith('.pth')) : continue
            path = Path(path, ckpt_name)
            inference_run(path, conf, i)
    else:
        inference_run(path, conf)


if __name__ == '__main__':
    main()
