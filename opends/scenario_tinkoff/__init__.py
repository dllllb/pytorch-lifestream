import os
import argparse
import json
import logging

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

import torch

from scenario_tinkoff.data import load_data, log_split_by_date, get_encoder, get_hist_count
from scenario_tinkoff.feature_preparation import load_user_features, load_item_features, COL_ID
from scenario_tinkoff.metrics import hit_rate_at_k, label_ranking_average_precision_score, ranking_score, \
    precision_at_k, tinkoff_reward
from scenario_tinkoff.models import StoriesRecModel, PopularModel, PairwiseMarginRankingLoss, ALSModel

logger = logging.getLogger(__name__)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=os.path.abspath,
                        default='../data/tinkoff')
    parser.add_argument('--embedding_file_name', default="embeddings.pickle")
    parser.add_argument('--train_size', type=float, default=0.75)

    parser.add_argument('--model_type', default='nn', choices=['nn'])

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_epoch', type=int, default=5)

    parser.add_argument('--optim_weight_decay', type=float, nargs='+', default=[0.0001])
    parser.add_argument('--optim_lr', type=float, default=0.01)
    parser.add_argument('--lr_step_size', type=int, default=1)
    parser.add_argument('--lr_step_gamma', type=float, default=0.5)

    parser.add_argument('--hidden_size', type=int, default=4)
    parser.add_argument('--user_layers', type=str)
    parser.add_argument('--item_layers', type=str)

    parser.add_argument('--use_user_popular_features', action='store_true')
    parser.add_argument('--use_trans_common_features', action='store_true')
    parser.add_argument('--use_gender', action='store_true')
    parser.add_argument('--use_trans_mcc_features', action='store_true')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--use_item_popular_features', action='store_true')

    parser.add_argument('--use_embedding_as_init', action='store_true')

    parser.add_argument('--loss', type=str, default='mse', choices=['mae', 'mse'])

    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--valid_batch_size', type=int, default=5000)
    parser.add_argument('--train_num_workers', type=int, default=8)
    parser.add_argument('--valid_num_workers', type=int, default=8)

    parser.add_argument('--exclude_seen_items', default=False)
    parser.add_argument('--check_random', action='store_true')
    parser.add_argument('--precision_k', type=int, default=10)

    parser.add_argument('--history_file', type=os.path.abspath, default='runs/scenario_tinkoff.json')
    parser.add_argument('--report_file', type=os.path.abspath, required=False)

    config = parser.parse_args(args)
    conf_str = json.dumps(vars(config), indent=2)
    logger.info(f'Config:\n{conf_str}')

    return config


def save_result(config, score, metrics):
    history_file = config.history_file

    if os.path.isfile(history_file) and os.stat(history_file).st_size > 0:
        with open(history_file, 'rt') as f:
            history = json.load(f)
    else:
        history = []

    history.append({
        'config': vars(config),
        'final_score': score,
        'metrics': metrics,
    })

    with open(history_file, 'wt') as f:
        json.dump(history, f, indent=2)


def convert_history_file(config):
    history_file = config.history_file
    report_file = config.report_file

    with open(history_file, 'rt') as f:
        history = json.load(f)

    df = json_normalize(history)

    metric_columns = [
        'final_score.train_reward', 'final_score.valid_reward',
        'final_score.valid_reward_at_0_sample',
        'final_score.valid_reward_at_2_sample',
        'final_score.valid_reward_at_8_sample',
    ]
    changing_columns = df.astype(str).nunique()[lambda x: x > 1].index.tolist()
    col_drop = metric_columns + ['metrics']
    columns = metric_columns + [col for col in changing_columns if col not in col_drop]
    df_results = df[columns]

    with pd.option_context(
        'display.float_format', '{:.4f}'.format,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
    ):
        logger.info(f'Results:\n{df_results}')
        with open(report_file, 'w') as f:
            print(df_results, file=f)


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


def main(config):
    if config.report_file is not None:
        return convert_history_file(config)

    if config.check_random:
        return check_random(config)

    device = torch.device(config.device)

    df_log = load_data(config)
    df_log_train, df_log_valid = log_split_by_date(df_log, config.train_size)

    df_train_hist_count = get_hist_count(df_log_train)
    df_log_valid = pd.merge(df_log_valid, df_train_hist_count, on=COL_ID, how='left')
    df_log_valid['hist_count'] = df_log_valid['hist_count'].fillna(0)

    df_users = load_user_features(config, df_log_train)
    df_items = load_item_features(config, df_log_train)

    if config.model_type == 'nn':
        user_encoder = get_encoder(df_log_train, 'customer_id', min_count=2)
        item_encoder = get_encoder(df_log_train, 'story_id')

        df_log_exclude = None

        model = StoriesRecModel(
            hidden_size=config.hidden_size,
            user_layers=config.user_layers,
            user_fixed_vector_size=0 if df_users is None else df_users.size,
            user_encoder=user_encoder,
            df_users=df_users,
            item_layers=config.item_layers,
            item_fixed_vector_size=0 if df_items is None else df_items.size,
            item_encoder=item_encoder,
            df_items=df_items,
            config=config,
            device=device,
        )

        def valid_fn():
            print('--- Inspect model: ---')
            for k, v in model.named_parameters():
                v = v.detach().cpu().numpy()
                print(f'{k:40}: ', end='')

                if v.size <= 10:
                    print(v)
                else:
                    values_str = [f'{p:7.3f}' for p in np.percentile(v, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])]
                    print('p: % {} %'.format(" ".join(values_str)))

            train_predict = model.model_predict(df_log_exclude, df_log_train)
            valid_predict = model.model_predict(df_log_exclude, df_log_valid)
            return {
                'train_reward': tinkoff_reward(train_predict),
                'valid_reward': tinkoff_reward(valid_predict),
                **{f'valid_reward_at_{int(k):01d}_sample': v
                   for k, v in tinkoff_reward(valid_predict, 'hist_count').to_dict().items()}
            }

        model.add_valid_fn(valid_fn)
    else:
        raise NotImplementedError(f'Not implemented for model_type: {config.model_type}')

    train_metrics = model.model_train(df_log=df_log_train)
    scores = train_metrics[-1]

    for k, v in scores.items():
        if k.endswith('_reward'):
            logger.info(f'{model.__class__.__name__} valid_predict {k}: {v:.4f}')

    save_result(config, scores, train_metrics)
