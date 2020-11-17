import logging

from dltranz.data_load.data_module.coles_data_module import ColesDataModuleTrain
import pytorch_lightning as pl
from dltranz.seq_mel import SequenceMetricLearning
from dltranz.util import get_conf

logger = logging.getLogger(__name__)

# if __name__ == '__main__':
#     switch_reproducibility_on()


def main(args=None):
    conf = get_conf(args)

    model = SequenceMetricLearning(conf['params'])
    dm = ColesDataModuleTrain(conf['data_module'])
    trainer = pl.Trainer(**conf['trainer'])
    trainer.fit(model, dm)

    if 'model_path' in conf:
        trainer.save_checkpoint(conf['model_path'], weights_only=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')
    logging.getLogger("lightning").setLevel(logging.INFO)

    logger.warning('    This script is not reproducible!')

    main()
