import logging
import sys

import torch
from sklearn.model_selection import StratifiedKFold

from dltranz.experiment import update_model_stats
from dltranz.metric_learn.inference_tools import infer_part_of_data, save_scores
from dltranz.metric_learn.ml_models import load_encoder_for_inference
from dltranz.seq_encoder import Squeeze
from dltranz.util import init_logger, get_conf
from scenario_x5.fit_target import create_ds, run_experiment, read_consumer_data

logger = logging.getLogger(__name__)


def load_model(params):
    pre_model = load_encoder_for_inference(params)
    trx_encoder = pre_model[0]
    rnn_encoder = pre_model[1]
    step_select_encoder = pre_model[2]

    model_type = params['model_type']
    if model_type in ('rnn', 'cpc_model'):
        input_size = params['rnn.hidden_size']
    elif model_type == 'transf':
        input_size = params['transf.input_size']
    else:
        raise NotImplementedError(f'NotImplementedError for model_type="{model_type}"')

    head_output_size = 4

    layers = [
        trx_encoder,
        rnn_encoder,
        step_select_encoder,
    ]
    if params['use_batch_norm']:
        layers.append(torch.nn.BatchNorm1d(input_size))

    layers.extend([
        torch.nn.Linear(input_size, head_output_size),
        torch.nn.LogSoftmax(dim=1),
    ])

    model = torch.nn.Sequential(*layers)

    return model


def prepare_parser(parser):
    pass


def main(_):
    init_logger('scenario_x5')
    init_logger('dltranz')
    init_logger('metric_learning')

    conf = get_conf(sys.argv[2:])

    model_f = load_model
    train_data = read_consumer_data(conf['dataset.train_path'], conf, is_train=True)
    test_data = read_consumer_data(conf['dataset.test_path'], conf, is_train=False)

    # train
    results = []

    skf = StratifiedKFold(conf['cv_n_split'])
    nrows = conf['params'].get('labeled_amount',-1) # semi-supervised setup. default = supervised

    target_values = [rec['target'] for rec in train_data]
    for i, (i_train, i_valid) in enumerate(skf.split(train_data, target_values)):
        logger.info(f'Train fold: {i}')
        i_train_data = [rec for i, rec in enumerate(train_data) if i in i_train]
        i_valid_data = [rec for i, rec in enumerate(train_data) if i in i_valid]

        if nrows > 0: i_train_data = i_train_data[:nrows]

        train_ds, valid_ds = create_ds(i_train_data, i_valid_data, conf)
        model, result = run_experiment(train_ds, valid_ds, conf['params'], model_f)

        # inference
        columns = conf['output.columns']
        train_scores = infer_part_of_data(i, i_valid_data, columns, model, conf)
        save_scores(train_scores, i, conf['output.valid'])

        test_scores = infer_part_of_data(i, test_data, columns, model, conf)
        save_scores(test_scores, i, conf['output.test'])

        results.append(result)

    # results
    stats_file = conf.get('stats.path', None)
    if stats_file is not None:
        update_model_stats(stats_file, conf, results)
