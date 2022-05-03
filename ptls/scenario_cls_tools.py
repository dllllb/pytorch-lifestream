import logging
import os
from collections import namedtuple
from functools import reduce
from glob import glob
from multiprocessing.pool import Pool
from operator import iadd

import lightgbm as lgb
import pandas as pd
import scipy.stats
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

import ptls.fastai.fastai_tools as fai
import ptls.neural_automl.neural_automl_tools as node

logger = logging.getLogger(__name__)


def prepare_common_parser(parser, data_path, output_file):
    parser.add_argument('--n_workers', type=int, default=5)
    parser.add_argument('--cv_n_split', type=int, default=5)
    parser.add_argument('--data_path', type=os.path.abspath, default=data_path)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--model_seed', type=int, default=42)
    parser.add_argument('--add_baselines', action='store_true')
    parser.add_argument('--add_emb_baselines', action='store_true')
    parser.add_argument('--baseline_name')
    parser.add_argument('--models', nargs='+', default=['linear', 'lgb', 'xgb'])
    parser.add_argument('--embedding_file_names', nargs='*', default=[])
    parser.add_argument('--score_file_names', nargs='*', default=[])
    parser.add_argument('--output_file', type=os.path.abspath, default=output_file)
    parser.add_argument('--labeled_amount', type=int, default=-1)


def read_train_test(data_path, dataset_file, test_ids_file, col_id):
    target = pd.read_csv(os.path.join(data_path, dataset_file))
    test_ids = set(pd.read_csv(os.path.join(data_path, test_ids_file))[col_id].tolist())

    ix_test = target[col_id].isin(test_ids)

    logger.info(f'Train size: {(~ix_test).sum()} clients')
    logger.info(f'Test size: {ix_test.sum()} clients')

    return target[~ix_test].set_index(col_id), target[ix_test].set_index(col_id)


def drop_na_target(col_target, df, df_name):
    ix = df[col_target].isna()
    logger.info(f'Drop {ix.sum()} rows with NA values from "{df_name}". There are {(~ix).sum()} rows left')
    return df[~ix]


def filter_infrequent_target(col_target, df, df_name, keep_values):
    ix = df[col_target].isin(keep_values)
    logger.info(f'Drop {(~ix).sum()} infrequent rows from "{df_name}". There are {ix.sum()} rows left')
    return df[ix]


def get_folds(df, col_target, cv_n_split, random_state, labeled_amount=-1):
    folds = []
    if labeled_amount < 0: labeled_amount = len(df)  # semi-supervised setup. default = supervised
    skf = StratifiedKFold(n_splits=cv_n_split, random_state=random_state, shuffle=True)
    for i_train, i_test in skf.split(df, df[col_target]):
        folds.append((
            df.iloc[i_train[:labeled_amount]],
            df.iloc[i_test]
        ))
    return folds


KWParamsTrainAndScore = namedtuple('KWParamsTrainAndScore', [
    'name',
    'fold_n',
    'load_features_f',
    'model_type',
    'model_params',
    'scorer_name',
    'scorer',
    'col_target',
    'df_train',
    'df_valid',
    'df_test',
])


def train_and_score(kw_params: KWParamsTrainAndScore):
    log_process_id = f'{kw_params.name:20}:{kw_params.model_type:6}:{kw_params.fold_n:2}'

    try:
        logger.info(f'[{log_process_id}] Started')

        features = kw_params.load_features_f()

        y_train = kw_params.df_train[kw_params.col_target]
        y_valid = kw_params.df_valid[kw_params.col_target]
        y_test = kw_params.df_test[kw_params.col_target]

        train_features = [df.reindex(index=kw_params.df_train.index) for df in features]
        valid_features = [df.reindex(index=kw_params.df_valid.index) for df in features]
        test_features = [df.reindex(index=kw_params.df_test.index) for df in features]

        df_fn = [df.quantile(0.5) for df in train_features]
        df_norm = [df.max() - df.min() + 1e-5 for df in train_features]

        X_train = pd.concat([df.fillna(fn) / n for df, fn, n in zip(train_features, df_fn, df_norm)], axis=1)
        X_valid = pd.concat([df.fillna(fn) / n for df, fn, n in zip(valid_features, df_fn, df_norm)], axis=1)
        X_test = pd.concat([df.fillna(fn) / n for df, fn, n in zip(test_features, df_fn, df_norm)], axis=1)

        if kw_params.model_type == 'linear':
            if kw_params.model_params.get('objective') == 'regression':
                model = LinearRegression(**{k: v for k, v in kw_params.model_params.items() if k != 'objective'})
            else:
                model = LogisticRegression(**{k: v for k, v in kw_params.model_params.items() if k != 'objective'})
        elif kw_params.model_type == 'xgb':
            if kw_params.model_params.get('objective', 'classification').startswith('reg'):
                model = xgb.XGBRegressor(**kw_params.model_params)
            else:
                model = xgb.XGBClassifier(**kw_params.model_params)
        elif kw_params.model_type == 'lgb':
            if kw_params.model_params.get('objective') == 'regression':
                model = lgb.LGBMRegressor(**kw_params.model_params)
            else:
                model = lgb.LGBMClassifier(**kw_params.model_params)
        elif kw_params.model_type == 'tabnet':
            if kw_params.model_params.get('objective') == 'regression':
                from pytorch_tabnet.tab_model import TabNetRegressor
                model = TabNetRegressor(**kw_params.model_params['model_params'])
            else:
                from pytorch_tabnet.tab_model import TabNetClassifier
                model = TabNetClassifier(**kw_params.model_params['model_params'])
        elif kw_params.model_type in ('neural_automl', 'fastai'):
            pass
        else:
            raise NotImplementedError(f'Unknown model type {kw_params.model_type}')

        if kw_params.model_type in ['linear', 'xgb', 'lgb']:
            model.fit(X_train, y_train)
            score_valid = kw_params.scorer(model, X_valid, y_valid)
            score_test = kw_params.scorer(model, X_test, y_test)
        elif kw_params.model_type == 'neural_automl':
            score_valid = node.train_from_config(X_train.values,
                                                 y_train.values.astype('long'),
                                                 X_valid.values,
                                                 y_valid.values.astype('long'),
                                                 kw_params.model_params)
            score_test = -1

        elif kw_params.model_type == 'fastai':
            score_valid = fai.train_from_config(X_train.values,
                                                y_train.values.astype('long'),
                                                X_valid.values,
                                                y_valid.values.astype('long'),
                                                kw_params.model_params)
            '''score_valid = fai.train_tabular(X_train.values, 
                                               y_train.values.astype('long'), 
                                               X_valid.values, 
                                               y_valid.values.astype('long'),
                                               kw_params.model_params)'''
            score_test = -1

        if kw_params.model_type == 'tabnet':
            model.fit(
                X_train.values,
                y_train.values,
                eval_set=[(X_valid.values, y_valid.values)],
                **kw_params.model_params['fit_params']
            )
            score_valid = kw_params.scorer(model, X_valid.values, y_valid.values)
            score_test = kw_params.scorer(model, X_test.values, y_test.values)

        logger.info(
            ' '.join([
                f'[{log_process_id}]',
                f'Finished with {kw_params.scorer_name}',
                f'valid={score_valid:.4f},',
                f'test={score_test:.4f}'
            ]))
    except Exception as ex:
        logging.exception(f'[{log_process_id}]: exception\n{ex}', exc_info=True)
        score_valid = None
        score_test = None

    res = {
        'name': '_'.join([kw_params.model_type, kw_params.name]),
        'fold_n': kw_params.fold_n,
        f'oof_{kw_params.scorer_name}': score_valid,
        f'test_{kw_params.scorer_name}': score_test,
    }
    return res


def train_and_score_mean_by_key(kw_params: KWParamsTrainAndScore):
    """
    The differences with `dltranz.scenario_cls_tools.train_and_score`:
    - Duplicated keys allowed in test_df and valid_df index
    - Probabilities from `model.predict` aggregates by `df.index` key with `mean` aggfunc
    - Aggregation is slightly different for roc_auc and accuracy metrics.
        This why you should choose metric and aggregation func manually

    :param kw_params:
    :return:
    """
    log_process_id = f'{kw_params.name:20}:{kw_params.model_type:6}:{kw_params.fold_n:2}'

    try:
        logger.info(f'[{log_process_id}] Started')

        features = kw_params.load_features_f()[0]

        train_ds = pd.merge(features, kw_params.df_train[[kw_params.col_target]], left_index=True, right_index=True)
        valid_ds = pd.merge(features, kw_params.df_valid[[kw_params.col_target]], left_index=True, right_index=True)
        test_ds = pd.merge(features, kw_params.df_test[[kw_params.col_target]], left_index=True, right_index=True)

        y_train = train_ds[kw_params.col_target]
        y_valid = valid_ds[kw_params.col_target]
        y_test = test_ds[kw_params.col_target]

        X_train = train_ds.drop(columns=kw_params.col_target)
        X_valid = valid_ds.drop(columns=kw_params.col_target)
        X_test = test_ds.drop(columns=kw_params.col_target)

        df_fn = X_train.quantile(0.5)
        df_norm = X_train.max() - X_train.min() + 1e-5

        X_train = X_train.fillna(df_fn) / df_norm
        X_valid = X_valid.fillna(df_fn) / df_norm
        X_test = X_test.fillna(df_fn) / df_norm

        if kw_params.model_type == 'linear':
            if kw_params.model_params.get('objective') == 'regression':
                model = LinearRegression(**{k:v for k,v in kw_params.model_params.items() if k != 'objective'})
            else:
                model = LogisticRegression(**{k:v for k,v in kw_params.model_params.items() if k != 'objective'})
        elif kw_params.model_type == 'xgb':
            if kw_params.model_params.get('objective', 'classification').startswith('reg'):
                model = xgb.XGBRegressor(**kw_params.model_params)
            else:
                model = xgb.XGBClassifier(**kw_params.model_params)
        elif kw_params.model_type == 'lgb':
            if kw_params.model_params.get('objective') == 'regression':
                model = lgb.LGBMRegressor(**kw_params.model_params)
            else:
                model = lgb.LGBMClassifier(**kw_params.model_params)
        elif kw_params.model_type in ('neural_automl', 'fastai'):
            pass
        else:
            raise NotImplementedError(f'Unknown model type {kw_params.model_type}')

        if kw_params.model_type in ['linear', 'xgb', 'lgb']:
            model.fit(X_train, y_train)

            def _score_roc_auc(model, df, y):
                df = df.copy()
                df['predict'] = model.predict_proba(df)[:, 1]
                df['y'] = y
                df = df.groupby(level=0).mean()
                return roc_auc_score(df['y'], df['predict'])

            def _score_accuracy(model, df, y):
                df = df.copy()
                predicts = pd.DataFrame(model.predict_proba(df), index=df.index)
                predicts = predicts.groupby(level=0).mean()

                df['y'] = y
                df = df.groupby(level=0).mean()
                return accuracy_score(df['y'], predicts.reindex(index=df.index).idxmax(axis=1))

            # TODO: Choose score function here manually
            score_valid = _score_accuracy(model, X_valid, y_valid)
            score_test = _score_accuracy(model, X_test, y_test)
        elif kw_params.model_type == 'neural_automl':
            raise NotImplementedError()

        elif kw_params.model_type == 'fastai':
            raise NotImplementedError()

        logger.info(
            ' '.join([
                f'[{log_process_id}]',
                f'Finished with {kw_params.scorer_name}',
                f'valid={score_valid:.4f},',
                f'test={score_test:.4f}'
            ]))
    except Exception as ex:
        logging.exception(f'[{log_process_id}]: exception\n{ex}', exc_info=True)
        score_valid = None
        score_test = None

    res = {
        'name': '_'.join([kw_params.model_type, kw_params.name]),
        'fold_n': kw_params.fold_n,
        f'oof_{kw_params.scorer_name}': score_valid,
        f'test_{kw_params.scorer_name}': score_test,
    }
    return res


def group_stat_results(df, group_col_name, col_agg_metric=None, col_list_metrics=None, eps=1e-12):
    def values(x):
        return '[' + ' '.join([f'{i:.3f}' for i in sorted(x)]) + ']'

    def t_interval(x, p=0.95):
        n = len(x)
        s = x.std(ddof=1)

        return scipy.stats.t.interval(p, n - 1, loc=x.mean(), scale=(s + eps) / (n ** 0.5))

    def t_int_l(x, p=0.95):
        return t_interval(x, p)[0]

    def t_int_h(x, p=0.95):
        return t_interval(x, p)[1]

    metric_aggregates = []
    metric_names = []
    if col_agg_metric is not None:
        metric_aggregates.extend([
            df.groupby(group_col_name)[m_col].agg(['mean', t_int_l, t_int_h, 'std', values])
            for m_col in col_agg_metric
        ])
        metric_names.extend(col_agg_metric)
    if col_list_metrics is not None:
        metric_aggregates.extend([
            df.groupby(group_col_name)[m_col].agg([values])
            for m_col in col_list_metrics
        ])
        metric_names.extend(col_list_metrics)

    df_results = pd.concat(metric_aggregates, axis=1, keys=metric_names).sort_index()
    return df_results


class WPool:
    def __init__(self, processes):
        self.processes = processes
        self.pool = Pool(processes=processes) if self.processes > 0 else None

    def map(self, func, iterable):
        if self.pool is not None:
            return self.pool.map(func, iterable)
        else:
            return reduce(iadd, map(func, iterable))

    def imap_unordered(self, func, iterable):
        if self.pool is not None:
            return self.pool.imap_unordered(func, iterable)
        else:
            return reduce(iadd, map(func, iterable))


def expand_path(data_path, wc_paths):
    data_path = os.path.join(data_path, '')  # ensure `/` at the end

    embedding_file_names = []
    for path in wc_paths:
        for n_path in glob(data_path + path):
            embedding_file_names.append(n_path[len(data_path):])
    logger.info(f'Found {len(embedding_file_names)} embedding files: [{embedding_file_names}]')
    return embedding_file_names
