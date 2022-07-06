import logging

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader

from ptls.data_load.utils import collate_feature_dict
from ptls.frames.inference_module import InferenceModule

logger = logging.getLogger(__name__)


def save_scores(df_scores, output_conf):
    # output
    output_name = output_conf.path
    output_format = output_conf.format
    if output_format not in ('pickle', 'csv', 'parquet'):
        logger.warning(f'Format "{output_format}" is not supported. Used default "pickle"')
        output_format = 'pickle'

    output_path = f'{output_name}.{output_format}'

    if output_format == 'pickle':
        df_scores.to_pickle(output_path)
    elif output_format == 'csv':
        df_scores.to_csv(output_path, sep=',', header=True, index=False)
    elif output_format == 'parquet':
        df_scores.to_parquet(output_path)
    else:
        raise AssertionError('Never happens')
    logger.info(f'{len(df_scores)} records saved to: "{output_path}"')


@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    model = InferenceModule(
        model=hydra.utils.instantiate(conf.inference.seq_encoder),
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

    gpus = 1 if torch.cuda.is_available() else 0
    gpus = conf.inference.get('gpus', gpus)
    df_scores = pl.Trainer(gpus=gpus, max_epochs=-1).predict(model, inference_dl)
    df_scores = pd.concat(df_scores, axis=0)
    logger.info(f'df_scores examples: {df_scores.shape}:')

    save_scores(df_scores, conf.inference.output)


if __name__ == '__main__':
    main()
