import os
from glob import glob

import numpy as np
import pandas as pd

from scenario_gender.const import DATASET_FILE, COL_ID


def df_norm(df):
    return df.div(df.sum(axis=1) + 1e-5, axis=0)


def _random(conf):
    all_clients = pd.read_csv(os.path.join(conf['data_path'], DATASET_FILE)).set_index(COL_ID)
    all_clients = all_clients.assign(random=np.random.rand(len(all_clients)))
    return all_clients[['random']]


def _client_agg(conf):
    df_transactions = pd.read_csv(os.path.join(conf['data_path'], 'transactions.csv'))

    agg_features = pd.concat([
        df_transactions.groupby(COL_ID)['amount'].agg(['sum', 'mean', 'std', 'min', 'max']),
        df_transactions.groupby(COL_ID)[['mcc_code', 'tr_type']].nunique(),
    ], axis=1)

    return agg_features


def _mcc_code_stat(conf):
    df_transactions = pd.read_csv(os.path.join(conf['data_path'], 'transactions.csv'))

    cat_counts_train = pd.concat([
        df_norm(df_transactions.pivot_table(
            index=COL_ID, columns='mcc_code', values='amount', aggfunc='count').fillna(0.0)),
        df_norm(df_transactions.pivot_table(
            index=COL_ID, columns='mcc_code', values='amount', aggfunc='mean').fillna(0.0)),
        df_norm(df_transactions.pivot_table(
            index=COL_ID, columns='mcc_code', values='amount', aggfunc='std').fillna(0.0)),
    ], axis=1, keys=['mcc_code_count', 'mcc_code_mean', 'mcc_code_std'])

    cat_counts_train.columns = ['_'.join(map(str, c)) for c in cat_counts_train.columns.values]
    return cat_counts_train


def _tr_type_stat(conf):
    df_transactions = pd.read_csv(os.path.join(conf['data_path'], 'transactions.csv'))

    cat_counts_train = pd.concat([
        df_norm(df_transactions.pivot_table(
            index=COL_ID, columns='tr_type', values='amount', aggfunc='count').fillna(0.0)),
        df_norm(df_transactions.pivot_table(
            index=COL_ID, columns='tr_type', values='amount', aggfunc='mean').fillna(0.0)),
        df_norm(df_transactions.pivot_table(
            index=COL_ID, columns='tr_type', values='amount', aggfunc='std').fillna(0.0)),
    ], axis=1, keys=['tr_type_count', 'tr_type_mean', 'tr_type_std'])

    cat_counts_train.columns = ['_'.join(map(str, c)) for c in cat_counts_train.columns.values]
    return cat_counts_train


def _metric_learning_embeddings(conf, file_name):
    df = pd.read_pickle(os.path.join(conf['data_path'], file_name)).set_index(COL_ID)
    return df


def load_features(
        conf,
        use_random=False,
        use_client_agg=False,
        use_mcc_code_stat=False,
        use_tr_type_stat=False,
        metric_learning_embedding_name=None
):
    features = []
    if use_random:
        features.append(_random(conf))

    if use_client_agg:
        features.append(_client_agg(conf))

    if use_mcc_code_stat:
        features.append(_mcc_code_stat(conf))

    if use_tr_type_stat:
        features.append(_tr_type_stat(conf))

    if metric_learning_embedding_name is not None:
        features.append(_metric_learning_embeddings(conf, metric_learning_embedding_name))

    return features


def load_scores(conf, target_scores_name):
    valid_files = glob(os.path.join(conf['data_path'], target_scores_name, 'valid', '*'))
    valid_scores = [pd.read_pickle(f).set_index(COL_ID) for f in valid_files]

    test_files = glob(os.path.join(conf['data_path'], target_scores_name, 'test', '*'))
    test_scores = [pd.read_pickle(f).set_index(COL_ID) for f in test_files]

    return valid_scores, test_scores
