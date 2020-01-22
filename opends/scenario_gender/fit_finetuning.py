import logging
import sys

import torch
from sklearn.model_selection import StratifiedKFold

from dltranz.experiment import update_model_stats
from dltranz.metric_learn.inference_tools import infer_part_of_data, save_scores
from dltranz.seq_encoder import Squeeze, LastStepEncoder
from dltranz.util import init_logger, get_conf
from scenario_gender.fit_target import create_ds, run_experiment, read_consumer_data

logger = logging.getLogger(__name__)


def load_model(conf):
    pretrained_model_path = conf['pretrained_model_path']

    pre_model = torch.load(pretrained_model_path)
    trx_encoder = pre_model[0]
    rnn_encoder = pre_model[1]

    input_size = conf['rnn.hidden_size']
    head_output_size = 1

    model = torch.nn.Sequential(
        trx_encoder,
        rnn_encoder,
        LastStepEncoder(),
        torch.nn.Linear(input_size, head_output_size),
        torch.nn.Sigmoid(),
        Squeeze(),
    )
    return model


def prepare_parser(parser):
    pass


def main(_):
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('metric_learning')

    conf = get_conf(sys.argv[2:])

    model_f = load_model
    train_data = read_consumer_data(conf)
    test_data = read_consumer_data(conf)

    # train
    results = []

    skf = StratifiedKFold(conf['cv_n_split'])
    target_values = [rec['target'] for rec in train_data]
    for i, (i_train, i_valid) in enumerate(skf.split(train_data, target_values)):
        logger.info(f'Train fold: {i}')
        i_train_data = [rec for i, rec in enumerate(train_data) if i in i_train]
        i_valid_data = [rec for i, rec in enumerate(train_data) if i in i_valid]

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
