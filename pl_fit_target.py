import logging

from dltranz.data_load.data_module.cls_data_module import ClsDataModuleTrain
import pytorch_lightning as pl

from dltranz.seq_cls import SequenceClassify
from dltranz.util import get_conf

logger = logging.getLogger(__name__)


def main(args=None):
    conf = get_conf(args)

    if 'seed_everything' in conf:
        pl.seed_everything(conf['seed_everything'])

    model = SequenceClassify(conf['params'])
    dm = ClsDataModuleTrain(conf['data_module'])
    trainer = pl.Trainer(**conf['trainer'])
    trainer.fit(model, dm)

    if 'model_path' in conf:
        trainer.save_checkpoint(conf['model_path'], weights_only=True)
        logger.info(f'Model weights saved to "{conf.model_path}"')

    train_accuracy = model.train_accuracy.compute()
    valid_accuracy = model.valid_accuracy.compute()
    trainer.test(test_dataloaders=dm.test_tataloader())
    test_accuracy = model.test_accuracy.compute()
    print(f'train: {train_accuracy:.4f}, valid: {valid_accuracy:.4f}, test: {test_accuracy:.4f}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')
    logging.getLogger("lightning").setLevel(logging.INFO)

    main()
