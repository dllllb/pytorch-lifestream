import logging
import torch.utils
import torch.utils.data
import yaml
import os
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ptls.data_load.utils import collate_feature_dict
from ptls.frames.inference_module import InferenceModule
from pathlib import Path
logger = logging.getLogger(__name__)

from fedcore.tools.ruler import PerformanceEvaluator

def collate_feature_dict_for_perf_eval(batch):
    return collate_feature_dict(batch), None

def save_scores(df_scores, output_path, additional='', output_format='csv'):
    # output
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

def eval_computational_metrics(model: torch.nn.Module, 
                               dataloader: torch.utils.data.DataLoader,
                               save_path:str,
                               i='',
                               n_batches=1):
    
    pev = PerformanceEvaluator(model, dataloader, n_batches=n_batches)
    d = pev.eval()
    with open(save_path, 'at+') as file:
        print(f'#{i}', *(f'{k}: {v}' for k, v in d.items()), file=file)  


@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)
    CKPT_FOLDER = conf.ckpts
    for i, ckpt_name in tqdm(enumerate(os.listdir(CKPT_FOLDER)), 'Checkpoint #'):
            if not ckpt_name.endswith('.pth'): continue
            path = Path(CKPT_FOLDER, ckpt_name)
            seq_encoder = torch.load(path).seq_encoder
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

            if 'n_batches_computational' in conf:
                eval_computational_metrics(
                    model, 
                    DataLoader(
                        dataset=dataset_inference,
                        collate_fn=collate_feature_dict_for_perf_eval,
                        shuffle=False,
                        num_workers=conf.inference.get('num_workers', 0),
                        batch_size=conf.inference.get('batch_size', 32),
                    ),
                    Path(CKPT_FOLDER, 'computational.txt'), 
                    i,
                    n_batches=conf.n_batches_computational
                )

            accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'

            LIMIT = conf.get('max_batch', None)
            df_scores = pl.Trainer(accelerator=accelerator, 
                                limit_predict_batches=LIMIT,
                                limit_test_batches=LIMIT,
                                limit_val_batches=LIMIT,
                                limit_train_batches=LIMIT,
                                max_epochs=-1).predict(model, inference_dl)
            df_scores = pd.concat(df_scores, axis=0)
            logger.info(f'df_scores examples: {df_scores.shape}:')

            # save_scores(df_scores, conf.inference.output, i)

if __name__ == '__main__':
    main()
