import logging
import torch

from agg_features_ts_preparation import load_agg_data
from dltranz.custom_layers import MLP
from dltranz.data_load.fast_tensor_data_loader import FastTensorDataLoader
from dltranz.experiment import update_model_stats
from dltranz.tabnet.tab_network import TabNet
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model, CheckpointHandler
from dltranz.util import init_logger, get_conf, switch_reproducibility_on

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def create_data_loaders(conf, load_data_function):
    data = load_data_function(conf)
    r = torch.randperm(len(data))
    data = data[r]

    valid_size = int(len(data) * conf['dataset.valid_size'])
    train_data, valid_data = torch.split(data, [len(data) - valid_size, valid_size])
    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    train_loader = FastTensorDataLoader(
        train_data, torch.zeros(len(train_data)),
        batch_size=conf['params.train.batch_size'],
        shuffle=True
    )

    valid_loader = FastTensorDataLoader(
        valid_data, torch.zeros(len(valid_data)),
        batch_size=conf['params.valid.batch_size']
    )

    return train_loader, valid_loader


def run_experiment(model, conf, train_loader, valid_loader):
    import time
    start = time.time()

    stats_file = conf['stats.path']
    params = conf['params']

    sampling_strategy = get_sampling_strategy(params)
    loss = get_loss(params, sampling_strategy)

    try:
        split_count = params['valid.split_strategy.split_count']
    except:
        split_count = conf['data_module.valid.split_strategy.split_count']

    valid_metric = {'BatchRecallTop': BatchRecallTop(k=split_count - 1)}
    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    train_handlers = []
    if 'checkpoints' in conf['params.train']:
        checkpoint = CheckpointHandler(
            model=model,
            **conf['params.train.checkpoints']
        )
        train_handlers.append(checkpoint)

    metric_values = fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metric,
                              train_handlers=train_handlers)

    exec_sec = time.time() - start

    if conf.get('save_model', False):
        save_dir = os.path.dirname(conf['model_path.model'])
        os.makedirs(save_dir, exist_ok=True)

        m_encoder = model[0] if conf['model_path.only_encoder'] else model

        torch.save(m_encoder, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')

    results = {
        'exec-sec': exec_sec,
        'Recall_top_K': metric_values,
    }

    if conf.get('log_results', True):
        update_model_stats(stats_file, params, results)


def main(args=None):
    conf = get_conf(args)

    train_loader, valid_loader = create_data_loaders(conf, load_data_function=load_agg_data)

    if conf['params.model_type'] == 'mlp':
        encoder = MLP(conf['dataset.features_count'], conf['params.mlp'])
    elif conf['params.model_type'] == 'tabnet':
        encoder = TabNet(conf['dataset.features_count'], **conf['params.tabnet'])
    else:
        raise AttributeError(f"Unknown model_type: {conf['params.model_type']}")

    run_experiment(train_loader, valid_loader, encoder, conf)

    if conf.get('save_model', False):
        agg_model = torch.load(conf['model_path.agg_model'])
        enc_agr_model = torch.nn.Sequential(agg_model, encoder)

        torch.save(enc_agr_model, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')

    main()
