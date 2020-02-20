import os
from glob import glob

import numpy as np
import pandas as pd

from experiments.scenario_tin_cls.const import DATASET_FILE, COL_ID


def df_norm(df):
    return df.div(df.sum(axis=1) + 1e-5, axis=0)


def _random(conf):
    all_clients = pd.read_csv(os.path.join(conf['data_path'], DATASET_FILE)).set_index(COL_ID)
    all_clients = all_clients.assign(random=np.random.rand(len(all_clients)))
    return all_clients[['random']]


def _metric_learning_embeddings(conf, file_name):
    df = pd.read_pickle(os.path.join(conf['data_path'], file_name)).set_index(COL_ID)
    return df


def _trans_common_features(conf):
    df_trans = pd.read_csv(os.path.join(conf['data_path'], 'transactions.csv'))
    df_amnt_agg = pd.concat([
        df_trans.groupby('customer_id')['transaction_amt'].agg(['count', 'sum', 'mean']),
        df_trans.groupby('customer_id')['transaction_amt'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack(),
    ], axis=1)
    df_amnt_agg = np.log1p(df_amnt_agg) / 10
    return df_amnt_agg


def _trans_mcc_features(conf):
    def norm_row(df):
        return df.div(df.sum(axis=1) + 1e-5, axis=0)

    df_trans = pd.read_csv(os.path.join(conf['data_path'], 'transactions.csv'))

    ix_mcc = df_trans['merchant_mcc'].value_counts().index.tolist()
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

    df_mcc_agg.columns = [f'mcc_col_{i}' for i, _ in enumerate(df_mcc_agg.columns)]
    return df_mcc_agg


def load_features(
        conf,
        use_random=False,
        use_trans_common_features=False,
        use_trans_mcc_features=False,
        metric_learning_embedding_name=None,
):
    features = []
    if use_random:
        features.append(_random(conf))

    if use_trans_common_features:
        features.append(_trans_common_features(conf))

    if use_trans_mcc_features:
        features.append(_trans_mcc_features(conf))

    if metric_learning_embedding_name is not None:
        features.append(_metric_learning_embeddings(conf, metric_learning_embedding_name))

    return features


def load_scores(conf, target_scores_name):
    valid_files = glob(os.path.join(conf['data_path'], target_scores_name, 'valid', '*'))
    valid_scores = [pd.read_pickle(f).set_index(COL_ID) for f in valid_files]

    test_files = glob(os.path.join(conf['data_path'], target_scores_name, 'test', '*'))
    test_scores = [pd.read_pickle(f).set_index(COL_ID) for f in test_files]

    return valid_scores, test_scores
