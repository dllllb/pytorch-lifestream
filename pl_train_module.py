import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dltranz.util import get_conf, get_cls

logger = logging.getLogger(__name__)


def main(args=None):
    conf = get_conf(args)

    if 'seed_everything' in conf:
        pl.seed_everything(conf['seed_everything'])

    data_module = get_cls(conf['params.data_module_class'])

    pl_module = get_cls(conf['params.pl_module_class'])

    model = pl_module(conf['params'])
    dm = data_module(conf['data_module'], model)

    _trainer_params = conf['trainer']
    _use_best_epoch = conf['params.train'].get('use_best_epoch', False)

    if _use_best_epoch:
        checkpoint_callback = ModelCheckpoint(monitor=model.metric_name, mode='max')
        logger.info(f'Create ModelCheckpoint callback with monitor="{model.metric_name}"')
        if 'callbacks' in _trainer_params:
            logger.warning(f'Overwrite `trainer.callbacks`, was "{_trainer_params.checkpoint_callback}". '
                           f'New value is ModelCheckpoint callback')
        _trainer_params['callbacks'] = [checkpoint_callback]

    if conf['params.train'].get('checkpoints_every_n_val_epochs', False):
        every_n_val_epochs = conf['params.train.checkpoints_every_n_val_epochs']
        checkpoint_callback = ModelCheckpoint(every_n_val_epochs=every_n_val_epochs, save_top_k=-1)
        logger.info(f'Create ModelCheckpoint callback every_n_val_epochs ="{every_n_val_epochs}"')
        if 'callbacks' in _trainer_params:
            logger.warning(f'Overwrite `trainer.callbacks`, was "{_trainer_params.checkpoint_callback}". '
                           f'New value is ModelCheckpoint callback')
        _trainer_params['callbacks'] = [checkpoint_callback]


    if 'logger_name' in conf:
        _trainer_params['logger'] = TensorBoardLogger(
            save_dir='lightning_logs',
            name=conf.get('logger_name'),
        )

    trainer = pl.Trainer(**_trainer_params)
    trainer.fit(model, dm)

    if 'model_path' in conf:
        if _use_best_epoch:
            from shutil import copyfile
            copyfile(checkpoint_callback.best_model_path, conf.model_path)
            logging.info(f'Best model stored in "{checkpoint_callback.best_model_path}" '
                         f'and copied to "{conf.model_path}"')
        else:
            trainer.save_checkpoint(conf['model_path'], weights_only=True)
            logger.info(f'Model weights saved to "{conf.model_path}"')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')
    logging.getLogger("lightning").setLevel(logging.INFO)

    main()
