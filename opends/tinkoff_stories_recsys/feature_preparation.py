import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureList:
    def __init__(self, feature_list):
        self.feature_list = feature_list

    @property
    def size(self):
        return sum((df.shape[1] for df in self.feature_list))

    def get_item(self, ix):
        features = []
        for df in self.feature_list:
            ix = ix if ix in df.index else -1
            features.append(df.loc[ix].values)
        return np.concatenate(features).astype(np.float32)


def get_user_reaction_proba(df_user_item):
    s_default_reaction_proba = df_user_item.groupby('event')['customer_id'].count() / len(df_user_item)

    df_user_reaction_proba = df_user_item \
        [lambda x: x['customer_id'].isin(
            df_user_item.groupby('customer_id')['event_dttm'].count()[lambda x: x > 3].index)] \
        .pivot_table(
        index='customer_id', columns='event',
        values='story_id', aggfunc='count', fill_value=0)
    df_user_reaction_proba = df_user_reaction_proba + s_default_reaction_proba
    df_user_reaction_proba = df_user_reaction_proba.div(df_user_reaction_proba.sum(axis=1), axis=0)

    df_user_reaction_proba = pd.concat([
        df_user_reaction_proba,
        s_default_reaction_proba.to_frame().T.assign(customer_id=-1).set_index('customer_id'),
    ], axis=0)

    logger.info(f'Prepared df_user_reaction_proba: {df_user_reaction_proba.shape}')
    return df_user_reaction_proba


def get_item_reaction_proba(df_user_item):
    s_default_reaction_proba = df_user_item.groupby('event')['customer_id'].count() / len(df_user_item)

    df_item_reaction_proba = df_user_item \
        [lambda x: x['story_id'].isin(
            df_user_item.groupby('story_id')['event_dttm'].count()[lambda x: x > 3].index)] \
        .pivot_table(
        index='story_id', columns='event',
        values='customer_id', aggfunc='count', fill_value=0)
    df_item_reaction_proba = df_item_reaction_proba + s_default_reaction_proba
    df_item_reaction_proba = df_item_reaction_proba.div(df_item_reaction_proba.sum(axis=1), axis=0)

    df_item_reaction_proba = pd.concat([
        df_item_reaction_proba,
        s_default_reaction_proba.to_frame().T.assign(story_id=-1).set_index('story_id'),
    ], axis=0)

    logger.info(f'Prepared df_item_reaction_proba: {df_item_reaction_proba.shape}')
    return df_item_reaction_proba


def get_trans_common_features(config):
    df_trans = pd.read_csv(os.path.join(config.data_path, 'transactions.csv'))
    df_amnt_agg = pd.concat([
        df_trans.groupby('customer_id')['transaction_amt'].agg(['count', 'sum', 'mean']),
        df_trans.groupby('customer_id')['transaction_amt'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack(),
    ], axis=1)
    df_amnt_agg = np.log1p(df_amnt_agg) / 10

    df_amnt_agg = pd.concat([
        df_amnt_agg,
        pd.DataFrame(data=np.zeros((1, df_amnt_agg.shape[1])), columns=df_amnt_agg.columns, index=[-1])
    ])

    logger.info(f'Prepared df_amnt_agg: {df_amnt_agg.shape}')
    return df_amnt_agg


def get_trans_mcc_features(config):
    def norm_row(df):
        return df.div(df.sum(axis=1) + 1e-5, axis=0)

    df_trans = pd.read_csv(os.path.join(config.data_path, 'transactions.csv'))

    ix_mcc = df_trans['merchant_mcc'].value_counts().iloc[:127].index.tolist()
    df_mcc_agg = df_trans \
        .assign(merchant_mcc=df_trans['merchant_mcc'].map({v: i + 1 for i, v in enumerate(ix_mcc)}).fillna(0))

    df_mcc_agg = pd.concat([
        norm_row(df_mcc_agg.pivot_table(index='customer_id', columns='merchant_mcc',
                                        values='transaction_amt', aggfunc='count').fillna(0)),
        norm_row(df_mcc_agg.pivot_table(index='customer_id', columns='merchant_mcc',
                                        values='transaction_amt', aggfunc='sum').fillna(0)),
        norm_row(df_mcc_agg.pivot_table(index='customer_id', columns='merchant_mcc',
                                        values='transaction_amt', aggfunc='std').fillna(0)),
    ], axis=1).fillna(0)

    df_mcc_agg = pd.concat([
        df_mcc_agg,
        pd.DataFrame(data=np.zeros((1, df_mcc_agg.shape[1])), columns=df_mcc_agg.columns, index=[-1])
    ])

    logger.info(f'Prepared df_mcc_agg: {df_mcc_agg.shape}')
    return df_mcc_agg


def get_embeddings(config):
    df_embeddings = pd.read_pickle(os.path.join(config.data_path, config.embedding_file_name))
    df_embeddings = df_embeddings.set_index('customer_id')

    df_embeddings = pd.concat([
        df_embeddings,
        pd.DataFrame(data=df_embeddings.mean().values.reshape(1, -1), columns=df_embeddings.columns, index=[-1])
    ])

    logger.info(f'Prepared df_embeddings: {df_embeddings.shape}')
    return df_embeddings


def load_user_features(config, df_user_item):
    feature_list = []

    if config.use_user_popular_features:
        feature_list.append(get_user_reaction_proba(df_user_item))

    if config.use_trans_common_features:
        feature_list.append(get_trans_common_features(config))

    if config.use_embeddings:
        feature_list.append(get_embeddings(config))

    if config.use_trans_mcc_features:
        feature_list.append(get_trans_mcc_features(config))

    if len(feature_list) == 0:
        return None
    return FeatureList(feature_list)


def load_item_features(config, df_user_item):
    feature_list = []

    # popular features
    if config.use_item_popular_features:
        feature_list.append(get_item_reaction_proba(df_user_item))

    if len(feature_list) == 0:
        return None
    return FeatureList(feature_list)
