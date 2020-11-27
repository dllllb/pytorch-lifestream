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
    dm = ClsDataModuleTrain(conf['data_module'], model)
    trainer = pl.Trainer(**conf['trainer'])
    trainer.fit(model, dm)

    if 'model_path' in conf:
        trainer.save_checkpoint(conf['model_path'], weights_only=True)
        logger.info(f'Model weights saved to "{conf.model_path}"')

    valid_metrics = {name: mf.compute() for name, mf in model.valid_metrics.items()}
    trainer.test(test_dataloaders=dm.test_dataloader(), ckpt_path=None, verbose=False)
    test_metrics = {name: mf.compute() for name, mf in model.test_metrics.items()}

    print(', '.join([f'valid_{name}: {v:.4f}' for name, v in valid_metrics.items()]))
    print(', '.join([f' test_{name}: {v:.4f}' for name, v in test_metrics.items()]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')
    logging.getLogger("lightning").setLevel(logging.INFO)

    main()
