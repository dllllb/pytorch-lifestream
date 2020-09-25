from dltranz.util import switch_reproducibility_on

if __name__ == '__main__':
    import sys
    sys.path.append('../')

import argparse
import json
import logging
import os
from copy import deepcopy

import numpy as np
import torch

from experiments.scenario_tinkoff import train_test_nn
from experiments.scenario_tinkoff import load_data, log_split_by_date
from experiments.scenario_tinkoff import history_file
from experiments.scenario_tinkoff import save_result

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def check_random(config):
    df_log = load_data(config)
    df_log_train, df_log_valid = log_split_by_date(df_log, config.train_size)

    predict = df_log_valid.assign(relevance=np.random.rand(len(df_log_valid)))
    scores = {
        'ranking_score': ranking_score(predict),
        'precision_at_k': precision_at_k(predict, k=config.precision_k),
    }

    for k, v in scores.items():
        logger.info(f'RandomRelevance predict {k}: {v:.4f}')

    save_result(config, scores, metrics=[])


def parse_args(args):
    parser = argparse.ArgumentParser()

    # common arguments
    parser.add_argument('--data_path', type=os.path.abspath, default='../data/tinkoff')
    parser.add_argument('--history_file', type=os.path.abspath, default='runs/scenario_tinkoff.json')

    subparsers = parser.add_subparsers()

    # train
    sub_parser = subparsers.add_parser('train')
    sub_parser.set_defaults(func=train_test_nn.main)
    train_test_nn.prepare_parser(sub_parser)

    # convert_history_file
    sub_parser = subparsers.add_parser('convert_history_file')
    sub_parser.set_defaults(func=history_file.main)
    history_file.prepare_parser(sub_parser)

    # check_random
    sub_parser = subparsers.add_parser('check_random')
    sub_parser.set_defaults(func=check_random)

    # parse
    config = parser.parse_args(args)
    conf_str = deepcopy(vars(config))
    del conf_str['func']
    conf_str = json.dumps(conf_str, indent=2)
    logger.info(f'Config:\n{conf_str}')

    return config


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-15s : %(message)s')

    np.set_printoptions(linewidth=160)

    config = parse_args(None)
    func = config.func
    delattr(config, 'func')
    func(config)
