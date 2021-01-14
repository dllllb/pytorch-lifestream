import logging
import torch

from functools import partial

from agg_features_ts_preparation import load_agg_data
from dltranz.custom_layers import MLP, TabularRowEncoder, EmbedderNetwork
from dltranz.data_load.fast_tensor_data_loader import FastTensorDataLoader
from dltranz.experiment import update_model_stats
from dltranz.loss import get_loss
from dltranz.tabnet.tab_network import TabNet, TabNetDecoder
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model, CheckpointHandler
from dltranz.util import init_logger, get_conf, switch_reproducibility_on

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def mask_batch(batch, lag=0, zero_prob=0.15):
    x, _ = batch
    mask = torch.ones((x.shape[0], x.shape[1] - lag, x.shape[2]), dtype=torch.int64)
    obf_vars = torch.bernoulli(mask * zero_prob).bool()

    if lag > 0:
        x = x[:, :-lag, :]
    masked_x = torch.mul(~obf_vars, x).reshape(-1, x.shape[-1]).contiguous()

    return masked_x, obf_vars.reshape(-1, x.shape[-1]).contiguous()


def create_masked_data_loaders(conf, load_data_function):
    data = load_data_function(conf)
    r = torch.randperm(len(data))
    data = data[r]

    valid_size = int(len(data) * conf['dataset.valid_size'])
    train_data, valid_data = torch.split(data, [len(data) - valid_size, valid_size])
    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    post_process_func = partial(
        mask_batch,
        lag=conf['params.train.target_lag'],
        zero_prob=conf['params.train.zero_prob']
    )

    train_loader = FastTensorDataLoader(
        train_data, torch.zeros(len(train_data)),
        batch_size=conf['params.train.batch_size'],
        shuffle=True,
        post_process_func=post_process_func
    )

    valid_loader = FastTensorDataLoader(
        valid_data, torch.zeros(len(valid_data)),
        batch_size=conf['params.valid.batch_size'],
        post_process_func=post_process_func
    )

    return train_loader, valid_loader


def get_model(conf):
    tabular_config = conf['params.tabular_encoder']

    tabular_row_encoder = TabularRowEncoder(
        input_dim=tabular_config['num_features_count'] + len(tabular_config['cat_features_dims']),
        cat_dims=tabular_config['cat_features_dims'],
        cat_idxs=[x + tabular_config['num_features_count'] for x in range(len(tabular_config['cat_features_dims']))],
        cat_emb_dim=tabular_config['cat_emb_dim']
    )

    if conf['params.model_type'] == 'mlp':
        encoder = MLP(tabular_row_encoder.output_size, conf['params.mlp'])
        conf['params.mlp_decoder']['hidden_layers_size'] += [tabular_row_encoder.output_size]
        decoder = MLP(encoder.output_size, conf['params.mlp_decoder'])
    elif conf['params.model_type'] == 'tabnet':
        encoder = TabNet(tabular_row_encoder.output_size, **conf['params.tabnet'])
        decoder = TabNetDecoder(tabular_row_encoder.output_size, **conf['params.tabnet'])
    else:
        raise AttributeError(f"Unknown model_type: {conf['params.model_type']}")

    network = torch.nn.Sequential(encoder, decoder)
    model = EmbedderNetwork(tabular_row_encoder, network)
    return model


def run_experiment(model, conf, train_loader, valid_loader):
    import time
    start = time.time()

    stats_file = conf['stats.path']
    params = conf['params']

    loss = get_loss(params)
    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    train_handlers = []
    if 'checkpoints' in conf['params.train']:
        checkpoint = CheckpointHandler(
            model=model,
            **conf['params.train.checkpoints']
        )
        train_handlers.append(checkpoint)

    metric_values = fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metrics={},
                              train_handlers=train_handlers)

    exec_sec = time.time() - start
    results = {
        'exec-sec': exec_sec,
        'Recall_top_K': metric_values,
    }

    if conf.get('log_results', True):
        update_model_stats(stats_file, params, results)


def main(args=None):
    conf = get_conf(args)

    train_loader, valid_loader = create_masked_data_loaders(conf, load_data_function=load_agg_data)
    model = get_model(conf)

    run_experiment(model, conf, train_loader, valid_loader)

    if conf.get('save_model', False):
        agg_model = torch.load(conf['model_path.agg_model'])

        tabular_row_encoder = model.embedder
        encoder = model.network[0]

        # aggregation model, tabular row encoder, encoder
        enc_agr_model = torch.nn.Sequential(agg_model, tabular_row_encoder, encoder)
        torch.save(enc_agr_model, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')

    main()
