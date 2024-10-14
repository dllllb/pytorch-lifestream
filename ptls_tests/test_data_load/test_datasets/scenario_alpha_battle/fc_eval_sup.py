import json
import logging
import torch

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from fedcore.api.utils.data import get_compression_input
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.api.main import FedCore
from torchmetrics import Accuracy, AUROC
from fc_setups import SETUPS
from tqdm import tqdm
from ptls.pl_inference import InferenceModule
from torch.utils.data.dataloader import DataLoader
import os
from ptls.data_load.utils import collate_feature_dict


logger = logging.getLogger(__name__)

def load_dm(conf):
    return hydra.utils.instantiate(conf.data_module)

def load_train(conf):
    model = hydra.utils.instantiate(conf.pl_module)
    model.seq_encoder.is_reduce_sequence = True
    if 'pretrained' in conf:
        model.seq_encoder = torch.load(conf.pretrained)

    dm = hydra.utils.instantiate(conf.data_module)

    comp_inp = get_compression_input(
            model, dm.train_dataloader(), dm.val_dataloader(),
            train_loss=lambda : model._loss if hasattr(model, '_loss') else None,
            task='classification', num_classes=2
        )
    type_ = conf.type_
    save_path = f'composition_results/{type_}'
    experiment_setup = SETUPS[type_]
    experiment_setup['output_folder'] = save_path
    if hasattr(conf, 'limit_train_batches'):
        experiment_setup['common']['max_train_batch'] = conf.limit_train_batches
    fedcore_compressor = FedCore(**experiment_setup)
    fedcore_compressor.fit((comp_inp, model), manually_done=True)
    return fedcore_compressor, dm


def fold_fit_test(conf, fold_id):
    conf['fold_id'] = fold_id

    fedcore_compressor, dm = load_train(conf)
    model = fedcore_compressor.optimised_model
    
    
    acc_mod = Accuracy('binary')
    aucroc_mod = AUROC('binary')

    for x, y in dm.test_dataloader():
        output = model(x.to(default_device()))
        print(output)
        acc_mod.update(output, y)
        aucroc_mod.update(output, y)

    print('ACC', acc_mod.compute())
    print('AUC', aucroc_mod.compute())


@hydra.main(version_base=None)
def main(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    CKPT_FOLDER = conf.ckpts
    dm = load_dm(conf)
    for i, ckpt_name in tqdm(enumerate(os.listdir(CKPT_FOLDER)), 'Checkpoint #'):
            if not ckpt_name.endswith('.pth'): continue
            
            path = Path(CKPT_FOLDER, ckpt_name)
            model = torch.load(path)
            
            acc_mod = Accuracy('binary')
            aucroc_mod = AUROC('binary')

            for x, y in dm.test_dataloader():
                print(y)
                output = model(x.to(default_device()))
                print(output)
                acc_mod.update(output, y)
                aucroc_mod.update(output, y)

 


if __name__ == '__main__':
    main()
