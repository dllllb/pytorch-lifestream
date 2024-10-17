import hydra
from fc_utils import fedcore_fit, get_experimental_setup
import logging
from pathlib import Path
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

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
    if hasattr(conf, 'limit_train_batches'):
        experiment_setup['common']['max_train_batch'] = conf.limit_train_batches
    if hasattr(conf, 'limit_valid_batches'):
        experiment_setup['common']['max_calib_batch'] = conf.limit_valid_batches
    if hasattr(conf, 'need_pretrain'):
        experiment_setup['need_pretrain'] = conf.need_pretrain
        
    # instantiate model
    model = hydra.utils.instantiate(conf.pl_module)
    if 'pretrained' in conf:
        model.seq_encoder = torch.load(conf.pretrained)

    # instantiate data_module
    dm = hydra.utils.instantiate(conf.data_module)

    # fitting
    fedcore_compressor = fedcore_fit(model, dm, experiment_setup)
    
    if 'save_encoder' in conf:
        torch.save(fedcore_compressor.optimised_model.seq_encoder, conf.save_encoder)
    return fedcore_compressor
        
if __name__ == '__main__':
    main()
