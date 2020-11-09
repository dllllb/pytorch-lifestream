import logging

from dltranz.data_load.data_module.coles_data_module import (
    ColesDataModuleTrain,
)
from dltranz.metric_learn.ml_models import ml_model_by_type
from dltranz.seq_mel import SequenceMetricLearning
from dltranz.util import get_conf
from metric_learning import run_experiment

logger = logging.getLogger(__name__)

# if __name__ == '__main__':
#     switch_reproducibility_on()


def main(args=None):
    conf = get_conf(args)

    model = SequenceMetricLearning(conf['params'])

    dm = ColesDataModuleTrain(conf['data_module'])
    dm.setup()
    train_loader, valid_loader = dm.train_dataloader(), dm.val_dataloader()

    return run_experiment(model.model, conf, train_loader, valid_loader)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    logger.warning('    This script is not reproducible!')

    main()
