import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    model = hydra.utils.instantiate(conf.pl_module)
    dm = hydra.utils.instantiate(conf.data_module)

    _trainer_params = conf.trainer
    _trainer_params_additional = {}
    _use_best_epoch = _trainer_params.get('use_best_epoch', False)

    if 'callbacks' in _trainer_params:
        logger.warning(f'Overwrite `trainer.callbacks`, was "{_trainer_params.checkpoint_callback}"')
    _trainer_params_callbacks = []

    if _use_best_epoch:
        checkpoint_callback = ModelCheckpoint(monitor=model.metric_name, mode='max')
        logger.info(f'Create ModelCheckpoint callback with monitor="{model.metric_name}"')
        _trainer_params_callbacks.append(checkpoint_callback)

    if _trainer_params.get('checkpoints_every_n_val_epochs', False):
        every_n_val_epochs = _trainer_params.checkpoints_every_n_val_epochs
        checkpoint_callback = ModelCheckpoint(every_n_epochs=every_n_val_epochs, save_top_k=-1)
        logger.info(f'Create ModelCheckpoint callback every_n_epochs ="{every_n_val_epochs}"')
        _trainer_params_callbacks.append(checkpoint_callback)

        if 'checkpoint_callback' in _trainer_params:
            del _trainer_params.checkpoint_callback
        if 'enable_checkpointing' in _trainer_params:
            del _trainer_params.enable_checkpointing
        del _trainer_params.checkpoints_every_n_val_epochs

    if 'logger_name' in conf:
        _trainer_params_additional['logger'] = TensorBoardLogger(
            save_dir='lightning_logs',
            name=conf.get('logger_name'),
        )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    _trainer_params_callbacks.append(lr_monitor)

    if len(_trainer_params_callbacks) > 0:
        _trainer_params_additional['callbacks'] = _trainer_params_callbacks
    trainer = pl.Trainer(**_trainer_params, **_trainer_params_additional)
    trainer.fit(model, dm)

    if 'model_path' in conf:
        if _use_best_epoch:
            # from shutil import copyfile
            # copyfile(checkpoint_callback.best_model_path, conf.model_path)
            model.load_from_checkpoint(checkpoint_callback.best_model_path)
            torch.save(model.seq_encoder, conf.model_path)
            logging.info(f'Best model stored in "{checkpoint_callback.best_model_path}" '
                         f'and copied to "{conf.model_path}"')
        else:
            torch.save(model.seq_encoder, conf.model_path)
            logger.info(f'Model weights saved to "{conf.model_path}"')


if __name__ == '__main__':
    main()
