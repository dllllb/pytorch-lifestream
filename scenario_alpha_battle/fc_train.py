import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from fedcore.api.utils.data import get_compression_input
from fedcore.api.main import FedCore
from functools import partial
from ptls.frames.coles.losses.contrastive_loss import ContrastiveLoss
from ptls.frames.coles.sampling_strategies.hard_negative_pair_selector import HardNegativePairSelector

import logging
import yaml
import os
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader

from ptls.data_load.utils import collate_feature_dict
from ptls.frames.inference_module import InferenceModule
from pathlib import Path
from utils import load_pkl
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles import CoLESModule
from ptls.nn import RnnEncoder, TrxEncoder, L2NormEncoder
from ptls.nn import L2NormEncoder
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


from fc_setups import SETUPS
logger = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)
        
    model = hydra.utils.instantiate(conf.pl_module)
    if 'pretrained' in conf:
        model.seq_encoder = torch.load(conf.pretrained)

    dm = hydra.utils.instantiate(conf.data_module)

    comp_inp = get_compression_input(
            model, dm.train_dataloader(), dm.val_dataloader(),
            train_loss=lambda : model._loss,
            task='classification', num_classes=2
        )
    type_ = conf.type_
    save_path = f'composition_results/{type_}'
    experiment_setup = SETUPS[type_]
    experiment_setup['output_folder'] = save_path
    if hasattr(conf, 'limit_train_batches'):
        experiment_setup['common']['max_train_batch'] = conf.limit_train_batches
    if hasattr(conf, 'limit_valid_batches'):
        experiment_setup['common']['max_calib_batch'] = conf.limit_valid_batches
    if hasattr(conf, 'need_pretrain'):
        experiment_setup['need_pretrain'] = conf.need_pretrain
    fedcore_compressor = FedCore(**experiment_setup)
    fedcore_compressor.fit((comp_inp, model), manually_done=True)
        
if __name__ == '__main__':
    main()
