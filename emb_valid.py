import os
import json
import logging
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger

from dltranz.data_load.data_module.emb_valid_data_module import EmbValidDataModule
import pytorch_lightning as pl

from dltranz.seq_to_target import SequenceToTarget
from dltranz.util import get_conf, get_cls


logger = logging.getLogger(__name__)


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        validation_frequency,
        test_frequency,
        model,
        dm,
        conf
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.validation_frequency = validation_frequency
        self.test_frequency = test_frequency
        self.model = model
        self.dm = dm
        self.conf = conf

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if 'model_path' in self.conf and global_step % self.save_step_frequency == 0:
            trainer.save_checkpoint(self.conf['model_path'], weights_only=True)
            logger.info(f'Model weights saved to "{self.conf.model_path}"')
        if global_step % self.validation_frequency == 0:
            trainer.test(test_dataloaders=self.dm.val_dataloader(), ckpt_path=None, verbose=False)
            valid_metrics = {name: float(mf.compute().item()) for name, mf in self.model.valid_metrics.items()}
            print(', '.join([f'valid_{name}: {v:.4f}' for name, v in valid_metrics.items()]))
            for name, v in valid_metrics.items():
                self.model.log(f'valid_{name}', v, prog_bar=True)
        if global_step % self.test_frequency == 0:
            trainer.test(test_dataloaders=self.dm.test_dataloader(), ckpt_path=None, verbose=False)
            test_metrics = {name: float(mf.compute().item()) for name, mf in self.model.test_metrics.items()}
            print(', '.join([f' test_{name}: {v:.4f}' for name, v in test_metrics.items()]))
            for name, v in test_metrics.items():
                self.model.log(f'test_{name}', v, prog_bar=True)


def main(args=None):
    conf = get_conf(args)

    if 'seed_everything' in conf:
        pl.seed_everything(conf['seed_everything'])

    if conf['params.encoder_type'] == 'pretrained':
        pre_conf = conf['params.pretrained']
        cls = get_cls(pre_conf['pl_module_class'])
        pretrained_module = cls.load_from_checkpoint(pre_conf['model_path'])
        pretrained_module.seq_encoder.is_reduce_sequence = True
    else:
        pretrained_module = None

    pretrained_encoder = None if pretrained_module is None else pretrained_module.seq_encoder
    model = SequenceToTarget(conf['params'], pretrained_encoder)
    dm = EmbValidDataModule(conf['data_module'], model)

    _trainer_params = conf['trainer']
    if 'logger_name' in conf:
        _trainer_params['logger'] = TensorBoardLogger(
            save_dir='lightning_logs',
            name=conf.get('logger_name'),
        )
    trainer = pl.Trainer(callbacks=[CheckpointEveryNSteps(50000, 500000, 50000, model, dm, conf)], **_trainer_params)
    trainer.fit(model, dm)

    valid_metrics = {name: float(mf.compute().item()) for name, mf in model.valid_metrics.items()}
    trainer.test(test_dataloaders=dm.test_dataloader(), ckpt_path=None, verbose=False)
    test_metrics = {name: float(mf.compute().item()) for name, mf in model.test_metrics.items()}

    print(', '.join([f'valid_{name}: {v:.4f}' for name, v in valid_metrics.items()]))
    print(', '.join([f' test_{name}: {v:.4f}' for name, v in test_metrics.items()]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')
    logging.getLogger("lightning").setLevel(logging.INFO)

    main()
