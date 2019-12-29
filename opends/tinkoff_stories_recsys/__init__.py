import os
import argparse
import json
import logging

import numpy as np
import torch

from tinkoff_stories_recsys.data import load_data, log_split_by_date, get_encoder
from tinkoff_stories_recsys.feature_preparation import load_user_features, load_item_features
from tinkoff_stories_recsys.metrics import hit_rate_at_k, label_ranking_average_precision_score, roc_auc_mc_score
from tinkoff_stories_recsys.models import StoriesRecModel, PopularModel, PairwiseMarginRankingLoss, ALSModel

logger = logging.getLogger(__name__)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=os.path.abspath,
                        default='/mnt/wind/data_open_ds/data-like-tinkoff-2019/')
    parser.add_argument('--embedding_file_name', default="tinkoff_all_vectors_large.pickle")
    parser.add_argument('--train_size', type=float, default=0.75)

    parser.add_argument('--model_type', default='nn', choices=['nn'])

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_epoch', type=int, default=5)

    parser.add_argument('--optim_weight_decay', type=float, default=0.0001)
    parser.add_argument('--optim_lr', type=float, default=0.01)

    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--user_one_for_all_size', type=int, default=0)
    parser.add_argument('--user_learn_embedding_size', type=int, default=0)
    parser.add_argument('--item_one_for_all_size', type=int, default=0)
    parser.add_argument('--item_learn_embedding_size', type=int, default=32)

    parser.add_argument('--use_user_popular_features', action='store_true')
    parser.add_argument('--use_trans_common_features', action='store_true')
    parser.add_argument('--use_trans_mcc_features', action='store_true')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--use_item_popular_features', action='store_true')

    parser.add_argument('--use_embedding_as_init', action='store_true')

    parser.add_argument('--loss', type=str, default='ranking', choices=['ranking'])
    parser.add_argument('--loss_margin', type=float, default=0.2)
    parser.add_argument('--nn_activation', type=str, default='sigmoid', choices=['cosine', 'sigmoid'])

    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--valid_batch_size', type=int, default=10000)
    parser.add_argument('--train_num_workers', type=int, default=16)
    parser.add_argument('--valid_num_workers', type=int, default=16)

    parser.add_argument('--exclude_seen_items', default=False)

    parser.add_argument('--history_file', type=os.path.abspath, default='tinkoff_stories_recsys_result.json')

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


def main(config):
    device = torch.device(config.device)

    df_log = load_data(config)
    df_log_train, df_log_valid = log_split_by_date(df_log, config.train_size)

    df_users = load_user_features(config, df_log_train)
    df_items = load_item_features(config, df_log_train)

    if config.model_type == 'nn':
        user_encoder = get_encoder(df_log_train, 'customer_id')
        item_encoder = get_encoder(df_log_train, 'story_id')

        df_log_exclude = None

        model = StoriesRecModel(
            hidden_size=config.hidden_size,
            activation=config.nn_activation,
            user_one_for_all_size=config.user_one_for_all_size,
            user_learn_embedding_size=config.user_learn_embedding_size,
            user_fixed_vector_size=0 if df_users is None else df_users.size,
            user_encoder=user_encoder,
            df_users=df_users,
            item_one_for_all_size=config.item_one_for_all_size,
            item_learn_embedding_size=config.item_learn_embedding_size,
            item_fixed_vector_size=0 if df_items is None else df_items.size,
            item_encoder=item_encoder,
            df_items=df_items,
            config=config,
            device=device,
        )

        def valid_fn():
            predict = model.model_predict(df_log_exclude, df_log_valid)
            return {
                'roc_auc_mc_score': roc_auc_mc_score(predict),
            }

        model.add_valid_fn(valid_fn)
    else:
        raise NotImplementedError(f'Not implemented for model_type: {config.model_type}')

    train_metrics = model.model_train(df_log=df_log_train)
    predict = model.model_predict(df_log_exclude, df_log_valid)

    scores = {
        'roc_auc_mc_score': roc_auc_mc_score(predict),
    }

    for k, v in scores.items():
        logger.info(f'{model.__class__.__name__} predict {k}: {v:.4f}')

    save_result(config, scores, train_metrics)
