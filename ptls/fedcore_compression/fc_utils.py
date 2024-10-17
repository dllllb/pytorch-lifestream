import json
import pickle

from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path
from typing import Union

from fedcore.api.utils.data import get_compression_input
from fedcore.api.main import FedCore
from fedcore.tools.ruler import PerformanceEvaluator
from fc_setups import SETUPS
from torch.nn.modules import Module
from torch.utils.data import DataLoader

# save & load
def save_json(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file)

def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)

def save_pkl(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_pkl(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# fedcore train
def fedcore_fit(model, dm, experiment_setup, loss=None):
    loss = loss or (lambda : getattr(model, 'loss', None))
    comp_inp = get_compression_input(
            model, dm.train_dataloader(), dm.val_dataloader(),
            train_loss=loss,
            task='classification', num_classes=2
        )
    fedcore_compressor = FedCore(**experiment_setup)
    fedcore_compressor.fit((comp_inp, model), manually_done=True)
    return fedcore_compressor

def extract_loss(model, conf=None):
        if hasattr(model, 'loss'):
            loss = lambda : getattr(model, 'loss')
        elif hasattr(model, '_loss'):
            loss = lambda : getattr(model, '_loss')
        elif conf is not None and 'loss' in conf:
            loss = lambda : instantiate(conf.loss)
        else:
            loss = None
        return loss

def eval_computational_metrics(model: Module, 
                               dataloader: DataLoader,
                               save_path:str,
                               id='',
                               n_batches=1):
    pev = PerformanceEvaluator(model, dataloader, n_batches=n_batches)
    d = pev.eval()
    with open(save_path, 'at+') as file:
        print(f'#{id}', *(f'{k}: {v}' for k, v in d.items()), file=file) 


def get_experimental_setup(name: Union[str, Path]):
    if name is None:
        return {}, 'default'
    if name in SETUPS:
        return SETUPS[name], name
    if not isinstance(name, Path):
        name = Path(name)
    return OmegaConf.to_container(OmegaConf.load(name)), name.name.split('.')[-2]
