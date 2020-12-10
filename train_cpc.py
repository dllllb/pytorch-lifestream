import logging
import torch
from ignite.metrics import RunningAverage, Loss

from dltranz.lightning_modules.cpc_module import CPC_Loss
from dltranz.experiment import CustomMetric, update_model_stats

from dltranz.metric_learn.ml_models import ml_model_by_type
from dltranz.seq_encoder.utils import LastStepEncoder
from dltranz.train import get_optimizer, get_lr_scheduler, CheckpointHandler, fit_model
from dltranz.util import init_logger, get_conf, switch_reproducibility_on
from dltranz.data_load import TrxDataset, ConvertingTrxDataset, SameTimeShuffleDataset, AllTimeShuffleDataset
from dltranz.data_load import create_train_loader, create_validation_loader
from metric_learning import prepare_data

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    switch_reproducibility_on()


def create_ds(train_data, valid_data, conf):
    train_ds = ConvertingTrxDataset(TrxDataset(train_data))
    valid_ds = ConvertingTrxDataset(TrxDataset(valid_data))

    return train_ds, valid_ds


def run_experiment(train_loader, valid_loader, model, conf):
    import time
    start = time.time()

    params = conf['params']

    loss = CPC_Loss(n_negatives=params['cpc.n_negatives'])

    valid_metric = {
        'loss': RunningAverage(Loss(loss)),
        'cpc accuracy': CustomMetric(lambda outputs, y: loss.cpc_accuracy(outputs, y))
    }

    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    train_handlers = []
    if 'checkpoints' in params['train']:
        checkpoint = CheckpointHandler(
            model=model,
            **params['train.checkpoints']
        )
        train_handlers.append(checkpoint)

    metric_values = fit_model(
        model,
        train_loader,
        valid_loader,
        loss,
        optimizer,
        scheduler,
        params,
        valid_metric,
        train_handlers=train_handlers)

    exec_sec = time.time() - start

    results = {
        'exec-sec': exec_sec,
        'metrics': metric_values,
    }

    stats_file = conf.get('stats.path', None)

    if stats_file is not None:
        update_model_stats(stats_file, params, results)
    else:
        return results


def main(args=None):
    conf = get_conf(args)

    train_data, valid_data = prepare_data(conf)
    train_ds, valid_ds = create_ds(train_data, valid_data, conf)
    if conf['params.train'].get('same_time_shuffle', False):
        train_ds = SameTimeShuffleDataset(train_ds)
        logger.info('SameTimeShuffle used')
    if conf['params.train'].get('all_time_shuffle', False):
        train_ds = AllTimeShuffleDataset(train_ds)
        logger.info('AllTimeShuffle used')

    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    cpc_e = ml_model_by_type(conf['params.model_type'])(conf['params'])

    train_loader = create_train_loader(train_ds, conf['params.train'])
    valid_loader = create_validation_loader(valid_ds, conf['params.valid'])
    run_experiment(train_loader, valid_loader, cpc_e, conf)

    if conf.get('save_model', False):
        trx_e, rnn_e = cpc_e.trx_encoder, cpc_e.seq_encoder
        l = LastStepEncoder()
        enc_agr_model = torch.nn.Sequential(trx_e, rnn_e, l)

        torch.save(enc_agr_model, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')

    main()
