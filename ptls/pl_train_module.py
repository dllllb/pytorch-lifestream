import hydra
import logging
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ptls.util import get_conf, get_cls
from pytorch_lightning.callbacks import LearningRateMonitor

logger = logging.getLogger(__name__)


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        model,
        dm,
        conf
    ):
        """
        Args:
            save_step_frequency: how often to save model in steps
        """
        self.conf = conf
        self.save_step_frequency = self.conf.get('params').get('save_step_frequency', 200)
        self.ckpts_path = self.conf.get('params').get('ckpts_path', 'ckpts3/')
        self.model = model
        self.dm = dm
        self.conf = conf

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if self.save_step_frequency and global_step % self.save_step_frequency == 0:
            trainer.save_checkpoint(self.ckpts_path + f'{global_step}')

@hydra.main()
def main(conf: DictConfig):
    OmegaConf.set_struct(conf, False)
    orig_cwd = hydra.utils.get_original_cwd()

    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    model = hydra.utils.instantiate(conf.pl_module)
    dm = hydra.utils.instantiate(conf.data_module, pl_module=model)

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
        checkpoint_callback = ModelCheckpoint(every_n_val_epochs=every_n_val_epochs, save_top_k=-1)
        logger.info(f'Create ModelCheckpoint callback every_n_val_epochs ="{every_n_val_epochs}"')
        _trainer_params_callbacks.append(checkpoint_callback)
        if 'checkpoint_callback' in _trainer_params:
            del _trainer_params.checkpoint_callback

    if 'logger_name' in conf:
        _trainer_params_additional['logger'] = TensorBoardLogger(
            save_dir='lightning_logs',
            name=conf.get('logger_name'),
        )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    _trainer_params_callbacks.append(lr_monitor)
    if 'ckpts_path' in conf:
        _trainer_params_callbacks.append(CheckpointEveryNSteps(model, dm, conf))

    if len(_trainer_params_callbacks) > 0:
        _trainer_params_additional['callbacks'] = _trainer_params_callbacks
    trainer = pl.Trainer(**_trainer_params, **_trainer_params_additional)
    trainer.fit(model, dm)

    if 'model_path' in conf:
        if _use_best_epoch:
            from shutil import copyfile
            copyfile(checkpoint_callback.best_model_path, conf.model_path)
            logging.info(f'Best model stored in "{checkpoint_callback.best_model_path}" '
                         f'and copied to "{conf.model_path}"')
        else:
            trainer.save_checkpoint(conf.model_path, weights_only=True)
            logger.info(f'Model weights saved to "{conf.model_path}"')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')
    logging.getLogger("lightning").setLevel(logging.INFO)

    main()
