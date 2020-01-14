import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _short_list(l):
    l = sorted(l)
    if len(l) <= 8:
        return l
    return l[:5] + ['...'] + l[-2:]


def log_dataset_info(df, desc):
    rec_total = len(df)
    users_total = df['user_id'].nunique()
    items_total = df['item_id'].nunique()
    item_per_user = df.groupby('user_id')['item_id'].count().mean()

    info = '\n'.join([
        f'  {rec_total} records',
        f'  {users_total} users: {_short_list(df["user_id"].unique())}',
        f'  {items_total} items: {_short_list(df["item_id"].unique())}',
        f'  {item_per_user:.2f} items_per_user',
    ])
    logger.info(f'{desc}:\n{info}')


def hit_rate_at_k(df_scores, df_true, desc=''):
    """

    :param df_scores: df: [user_id, item_id, proba]
    :param df_true:  [user_id, item_id]
    :param k:
    :return:
    """
    log_dataset_info(df_scores, f'df_scores {desc} in hit_rate')
    log_dataset_info(df_true, f'df_true {desc} in hit_rate')

    df = pd.merge(df_scores, df_true, how='inner', on=['user_id', 'item_id'])
    users_hit = df['user_id'].nunique()
    users_total = df_scores['user_id'].nunique()
    logger.info(f'recs: {len(df_scores)}, test: {len(df_true)}, df: {len(df)}, '
                f'user_hit: {users_hit}, user_total: {users_total}')

    u_users_r = set(df_scores['user_id'].tolist())
    u_users_t = set(df_true['user_id'].tolist())
    logger.info(f'u_users_r: {len(u_users_r)}, u_users_t: {len(u_users_t)}, '
                f'u_users_r & t: {len(u_users_r.intersection(u_users_t))}')

    u_items_r = set(df_scores['item_id'].tolist())
    u_items_t = set(df_true['item_id'].tolist())
    logger.info(f'u_items_r: {len(u_items_r)}, u_items_t: {len(u_items_t)}, '
                f'u_items_r & t: {len(u_items_r.intersection(u_items_t))}')

    return float(users_hit / users_total)


def label_ranking_average_precision_score(df_scores, df_true, reduce=True, desc=''):
    log_dataset_info(df_scores, f'df_scores {desc} in LRAP')
    log_dataset_info(df_true, f'df_true {desc} in LRAP')

    df = pd.merge(df_scores, df_true.assign(hit=1), how='left', on=['user_id', 'item_id'])
    df['hit'] = df['hit'].fillna(0)
    df = df.sort_values(['user_id', 'relevance'], ascending=[True, False])

    df['rank'] = df.groupby('user_id').cumcount() + 1
    df['hit_count'] = df.groupby('user_id')['hit'].cumsum()
    df['score'] = df['hit_count'] / df['rank']
    df = df[df['hit'].eq(1)]

    rank_hist = df.groupby('rank')['user_id'].count().sort_index()
    total_rank_hist = rank_hist.sum()
    show_pos = 5
    info = ', '.join(
        f'{x:.2f}' for x in [v / total_rank_hist for _, v in sorted(rank_hist.iloc[:show_pos].to_dict().items())] +
        [rank_hist.iloc[show_pos:].sum() / total_rank_hist]
    )
    logger.info(f'rank_hist: [{info} ...] from {total_rank_hist}')

    df = df.groupby('user_id')['score'].mean()

    if not reduce:
        return df
    score = df.mean()
    return float(score)


def ranking_score(df):
    def pair_ranking_rate(df):
        events = df['event'].map({'dislike': 0, 'skip': 1, 'view': 2, 'like': 3}).values
        scores = df['relevance'].values
        mask = (np.sign(events.reshape(-1, 1) - events.reshape(1, -1)) > 0).astype(int)
        right_pairs = ((mask * np.sign(scores.reshape(-1, 1) - scores.reshape(1, -1))) > 0).astype(float)
        return right_pairs.sum() / mask.sum() if mask.sum() > 0 else np.NaN

    return df.groupby('customer_id').apply(pair_ranking_rate).mean()
