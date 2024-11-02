"""Module for training and compression using FedCore"""
import logging

import hydra
from omegaconf import DictConfig
from pathlib import Path
import pytorch_lightning as pl
import torch

from ptls.fedcore_compression.fc_utils import fedcore_fit, get_experimental_setup, extract_loss


logger = logging.getLogger(__name__)

@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    # prepare some experiment setup
    setup = conf.get('setup', None)
    experiment_setup, setup_name = get_experimental_setup(setup)
    save_path = f'composition_results/{setup_name}'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    experiment_setup['output_folder'] = save_path
    if 'common' not in experiment_setup:
        experiment_setup['common'] = {}
    experiment_setup['common']['batch_limit'] = conf.get('limit_train_batches', None)
    experiment_setup['common']['calib_batch_limit'] = conf.get('limit_valid_batches', None)
    experiment_setup['need_fedot_pretrain'] = conf.get('need_fedot_pretrain', False)
    experiment_setup['need_evo_opt'] = conf.get('need_evo_opt', False)
    experiment_setup['distributed_compression'] = conf.get('distributed_compression', False)
        
    # instantiate model
    model = hydra.utils.instantiate(conf.pl_module)
    if 'pretrained' in conf and conf.pretrained:
        model.seq_encoder = torch.load(conf.pretrained)

    # instantiate data_module
    dm = hydra.utils.instantiate(conf.data_module)

    # fitting
    fedcore_compressor = fedcore_fit(model, dm, experiment_setup, loss=extract_loss(model, conf))
    
    if 'save_encoder' in conf:
        torch.save((fedcore_compressor.optimised_model.seq_encoder
                        if hasattr(fedcore_compressor.optimised_model, 'seq_encoder')
                        else fedcore_compressor.optimised_model),
                    conf.save_encoder)
    return fedcore_compressor
        
if __name__ == '__main__':
    main()
