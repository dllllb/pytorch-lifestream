"""Functions that integrate FedCore training process into pytorch-lifestream"""
import json
import pickle

from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path
from typing import Union, Callable, Optional, Any

from fedcore.api.utils.data import get_compression_input
from fedcore.api.main import FedCore
from fedcore.tools.ruler import PerformanceEvaluator
from pytorch_lightning import LightningDataModule
from torch.nn.modules import Module
from torch.utils.data import DataLoader

from ptls.fedcore_compression.fc_setups import SETUPS

# save & load
def save_json(obj: Any, path: Union[str, Path]):
    with open(path, 'w') as file:
        json.dump(obj, file)

def load_json(path: Union[str, Path]) -> Any:
    with open(path, 'r') as file:
        return json.load(file)

def save_pkl(obj:Any, path: Union[str, Path]):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_pkl(path: Union[str, Path]) -> Any:
    with open(path, 'rb') as file:
        return pickle.load(file)

# fedcore train
def fedcore_fit(model: Module, 
                dm: LightningDataModule, 
                experiment_setup: dict, 
                loss: Optional[Callable]=None, 
                n_cls: Optional[int] = None, 
                **kwargs) -> FedCore:
    """
    Fits a FedCore model using the provided data manager and experiment setup.

    Parameters:
        model (Model): model to be trained
        dm (PTLS Data Module):  that provides training and validation data loaders.
        experiment_setup (dict): configuration for the FedCore compressor.
        loss (callable): Optional lambda function or value representing the loss function. If not provided, it defaults to the loss attribute of the model.

    Returns:
    - FedCore instance fitted with the compressed input data and the model.
    """
    loss = loss or (lambda : getattr(model, 'loss', None))
    comp_inp = get_compression_input(
            model, dm.train_dataloader(), dm.val_dataloader(),
            train_loss=loss,
            task='classification', num_classes=n_cls
        )
    fedcore_compressor = FedCore(**experiment_setup)
    fedcore_compressor.fit((comp_inp, model), manually_done=True)
    return fedcore_compressor

def extract_loss(model: Module, conf: Optional[dict] = None) -> Optional[Callable]:
    """
    Extracts the loss function from a given model or configuration.
    Parameters:
      model: The model object having `loss` or `_loss` specified.
      conf: An optional configuration object that may contain a loss specification if the model does not provide one.

    Returns:
     A callable that returns the loss function if found; otherwise, it returns None.

    Notes:
    - If the model has a loss attribute, that will be used.
    - If the model does not have a loss attribute but has a _loss attribute, that will be used instead.
    - If both attributes are missing, and conf is provided with a loss, it will instantiate and return that loss.
    - If none of these conditions are met, the function will return None.
    """
    
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
                               save_path:  str,
                               id: Any = '',
                               n_batches: int = 1,
                               device: Optional[str] = None) -> dict:
    """
    Evaluates the computational metrics (latency, throughput) and size of the model using the provided DataLoader
    This function adresses to FedCore PerformanceEvaluator.

    Args:
        model (Module): The neural network model to be evaluated.
        dataloader (DataLoader): The DataLoader providing the input data for evaluation.
        save_path (str): The file path where the evaluation results will be saved.
        id (Any, optional): An optional identifier for this evaluation run (default is an empty string).
        n_batches (int, optional): The number of batches to evaluate for the metrics (default is 1).

    Returns:
        dict: A dictionary containing the computed metrics, which may include:
            - latency: Estimated latency of the model in milliseconds.
            - throughput: Estimated throughput of the model in samples per second.
            - size: Estimated size of model's weights and buffers in MB.
    
    Note:
        The results are appended to the specified file in a human-readable format, allowing for easy tracking 
        of performance metrics over time.
    """
    pev = PerformanceEvaluator(model, dataloader, device=device, n_batches=n_batches)
    d = pev.eval()
    with open(save_path, 'at+') as file:
        print(f'#{id}', *(f'{k}: {v}' for k, v in d.items()), file=file)
    return d 


def get_experimental_setup(name: Union[str, Path]) -> dict:
    """
    Retrieves the experimental setup based on the provided name.

    Parameters:
      name: A string or Path representing the name of the experimental setup. 

    Returns:
    A tuple containing:
    - A dictionary with the experimental setup configuration.
    - The name of the setup as a string. If the name was None, it returns 'default'.

    Behavior:
    - If the name is found in the predefined SETUPS, it returns the corresponding setup.
    - If name is not a Path, it is converted to a Path object.
    - The function loads the configuration from a file if the name is a path, converting it to a container using OmegaConf and returning its name without the file extension.
    """
    if name is None:
        return {}, 'default'
    if name in SETUPS:
        return SETUPS[name], name
    if not isinstance(name, Path):
        name = Path(name)
    d = OmegaConf.to_container(OmegaConf.load(name))
    return d, name.name.split('.')[-2]
