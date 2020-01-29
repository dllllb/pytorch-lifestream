import logging
from functools import reduce
from multiprocessing import Pool
from operator import iadd

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from dltranz.neural_automl.neural_automl_tools import train_from_config
from dltranz.scenario_cls_tools import prepare_common_parser, read_train_test, get_folds, group_stat_results
from scenario_age_pred.const import (
    DEFAULT_DATA_PATH, DEFAULT_RESULT_FILE, TEST_IDS_FILE, DATASET_FILE, COL_ID, COL_TARGET,
)
from scenario_age_pred.features import load_features, load_scores

logger = logging.getLogger(__name__)


def prepare_parser(parser):
    return prepare_common_parser(parser, data_path=DEFAULT_DATA_PATH, output_file=DEFAULT_RESULT_FILE)


def train_and_score(args):
    name, fold_n, conf, params, model_type, train_target, valid_target, test_target = args

    logger.info(f'[{name}:{fold_n}] Started: {params}')

    features = load_features(conf, **params)

    y_train = train_target[COL_TARGET]
    y_valid = valid_target[COL_TARGET]
    y_test = test_target[COL_TARGET]

    X_train = pd.concat([df.reindex(index=train_target.index) for df in features], axis=1)
    X_valid = pd.concat([df.reindex(index=valid_target.index) for df in features], axis=1)
    X_test = pd.concat([df.reindex(index=test_target.index) for df in features], axis=1)

    if model_type == 'xgb':
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            n_jobs=4,
            seed=conf['model_seed'],
            n_estimators=300)
    elif model_type == 'lgb':
        model = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='multiclass',
            num_class=4,
            metric='multi_error',
            n_estimators=1000,
            learning_rate=0.02,
            subsample=0.75,
            subsample_freq=1,
            feature_fraction=0.75,
            max_depth=12,
            lambda_l1=1,
            lambda_l2=1,
            min_data_in_leaf=50,
            num_leaves=50,
            random_state=conf['model_seed'],
            n_jobs=4)
    elif model_type == 'neural_automl':
        pass
    else:
        raise NotImplementedError(f'unknown model type {model_type}')

    if model_type != 'neural_automl':
        model.fit(X_train, y_train)
        valid_accuracy = (y_valid == model.predict(X_valid)).mean()
        test_accuracy = (y_test == model.predict(X_test)).mean()
    else:
        valid_accuracy = train_from_config(X_train.values,
                                           y_train.values.astype('long'),
                                           X_valid.values,
                                           y_valid.values.astype('long'),
                                           'age.json')

    logger.info(
        f'[{name}:{fold_n}] Finished with accuracy valid={valid_accuracy:.4f}, test={test_accuracy:.4f}: {params}')

    res = {}
    res['name'] = '_'.join([model_type, name])
    res['model_type'] = model_type
    res['fold_n'] = fold_n
    res['oof_accuracy'] = valid_accuracy
    res['test_accuracy'] = test_accuracy
    return res


def get_scores(args):
    name, conf, params, df_target, test_target = args

    logger.info(f'[{name}] Scoring started: {params}')

    result = []
    valid_scores, test_scores = load_scores(conf, **params)
    for fold_n, (valid_fold, test_fold) in enumerate(zip(valid_scores, test_scores)):
        valid_fold['pred'] = np.argmax(valid_fold.values, 1)
        test_fold['pred'] = np.argmax(test_fold.values, 1)
        valid_fold = valid_fold.merge(df_target, on=COL_ID, how='left')
        test_fold = test_fold.merge(test_target, on=COL_ID, how='left')

        result.append({
            'name': name,
            'fold_n': fold_n,
            'oof_accuracy': (valid_fold['pred'] == valid_fold[COL_TARGET]).mean(),
            'test_accuracy': (test_fold['pred'] == test_fold[COL_TARGET]).mean(),
        })

    return result


def main(conf):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    approaches_to_train = {
        'baseline': {'use_client_agg': True, 'use_small_group_stat': True},
        **{
            f"embeds: {file_name}": {'metric_learning_embedding_name': file_name}
            for file_name in conf['ml_embedding_file_names']
        }
    }

    approaches_to_score = {
        f"scores: {file_name}": {'target_scores_name': file_name}
        for file_name in conf['target_score_file_names']
    }

    df_target, test_target = read_train_test(conf['data_path'], DATASET_FILE, TEST_IDS_FILE, COL_ID)
    folds = get_folds(df_target, COL_TARGET, conf['cv_n_split'], conf['random_state'])

    args_list = [(name, fold_n, conf, params, model_type, train_target, valid_target, test_target)
                 for name, params in approaches_to_train.items()
                 for fold_n, (train_target, valid_target) in enumerate(folds)
                 for model_type in ['xgb', 'lgb']
                 ]

    pool = Pool(processes=conf['n_workers'])
    results = pool.map(train_and_score, args_list)
    df_results = pd.DataFrame(results).set_index('name')[['oof_accuracy', 'test_accuracy']]

    # score already trained models on valid and test sets
    pool = Pool(processes=conf['n_workers'])
    args_list = [(name, conf, params, df_target, test_target) for name, params in approaches_to_score.items()]
    results = reduce(iadd, pool.map(get_scores, args_list))
    df_scores = pd.DataFrame(results).set_index('name')[['oof_accuracy', 'test_accuracy']]

    # combine results
    df_results = pd.concat([df_results, df_scores])
    df_results = group_stat_results(df_results, 'name', ['oof_accuracy', 'test_accuracy'])

    with pd.option_context(
            'display.float_format', '{:.4f}'.format,
            'display.max_columns', None,
            'display.expand_frame_repr', False,
    ):
        logger.info(f'Results:\n{df_results}')
        with open(conf['output_file'], 'w') as f:
            print(df_results, file=f)
