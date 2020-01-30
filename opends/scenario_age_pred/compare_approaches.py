if __name__ == '__main__':
    import sys
    sys.path.append('../')

import logging
import os
from multiprocessing import Pool

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from functools import reduce
from operator import iadd

from dltranz.util import group_stat_results
from scenario_age_pred.features import load_features, load_scores
import dltranz.neural_automl.neural_automl_tools as node
import dltranz.fastai.fastai_tools as fai

logger = logging.getLogger(__name__)


def prepare_parser(parser):
    parser.add_argument('--n_workers', type=int, default=5)
    parser.add_argument('--cv_n_split', type=int, default=5)
    parser.add_argument('--data_path', type=os.path.abspath, default='../data/age-pred/')
    parser.add_argument('--test_size', type=float, default=0.4)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--model_seed', type=int, default=42)
    parser.add_argument('--ml_embedding_file_names', nargs='+', default=['embeddings.pickle'])
    parser.add_argument('--target_score_file_names', nargs='+', default=['target_scores', 'finetuning_scores'])
    parser.add_argument('--output_file', type=os.path.abspath, default='runs/scenario_age_pred.csv')
    parser.add_argument('--pool', type=bool, default=False)


def read_target(conf):
    target = pd.read_csv(os.path.join(conf['data_path'], 'train_target.csv'))
    test_ids = set(pd.read_csv(os.path.join(conf['data_path'], 'test_ids.csv'))['client_id'].tolist())
    is_test = [(x in test_ids) for x in target['client_id']]

    logger.info(f'Train size: {len(target) - sum(is_test)} clients')
    logger.info(f'Test size: {sum(is_test)} clients')

    return target[[not x for x in is_test]].set_index('client_id'), target[is_test].set_index('client_id')


def train_and_score(args):
    name, fold_n, conf, params, model_type, train_target, valid_target, test_target = args

    logger.info(f'[{name}:{fold_n}] Started: {params}')

    features = load_features(conf, **params)

    y_train = train_target['bins']
    y_valid = valid_target['bins']
    y_test = test_target['bins']

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
    elif model_type not in ['neural_automl', 'fastai']:
        raise NotImplementedError(f'unknown model type {model_type}')

    if model_type in ['xgb', 'lgb']:
        model.fit(X_train, y_train)
        valid_accuracy = (y_valid == model.predict(X_valid)).mean()
        test_accuracy = (y_test == model.predict(X_test)).mean()
    elif model_type == 'neural_automl':
        valid_accuracy = node.train_from_config(X_train.values, 
                                                y_train.values.astype('long'), 
                                                X_valid.values, 
                                                y_valid.values.astype('long'),
                                                'age.json')
        test_accuracy = -1
    elif model_type == 'fastai':
        valid_accuracy = fai.train_from_config(X_train.values, 
                                               y_train.values.astype('long'), 
                                               X_valid.values, 
                                               y_valid.values.astype('long'),
                                               'age_tabular.json')
        '''valid_accuracy = fai.train_tabular(X_train.values, 
                                               y_train.values.astype('long'), 
                                               X_valid.values, 
                                               y_valid.values.astype('long'),
                                               'age_tabular.json')'''
        test_accuracy = -1

    logger.info(f'[{name}:{fold_n}] Finished with accuracy valid={valid_accuracy:.4f}, test={test_accuracy:.4f}: {params}')

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
        valid_fold = valid_fold.merge(df_target, on='client_id', how = 'left')
        test_fold = test_fold.merge(test_target, on='client_id', how = 'left')

        result.append({
            'name' : name,
            'fold_n' : fold_n,
            'oof_accuracy' : (valid_fold['pred'] == valid_fold['bins']).mean(),
            'test_accuracy' : (test_fold['pred'] == test_fold['bins']).mean(),
        })

    return result


def main(conf):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    approaches_to_train = {
        #'baseline' : {'use_client_agg': True, 'use_small_group_stat': True},
        **{
            f"embeds: {file_name}" : {'metric_learning_embedding_name': file_name} for file_name in conf['ml_embedding_file_names']
        }
    }

    approaches_to_score = {
        #f"scores: {file_name}" : {'target_scores_name': file_name} for file_name in conf['target_score_file_names']
    }

    df_target, test_target  = read_target(conf)


    # train model on features and score valid and test sets
    folds = []
    skf = StratifiedKFold(n_splits=conf['cv_n_split'], random_state=conf['random_state'], shuffle=True)
    for i_train, i_test in skf.split(df_target, df_target['bins']):
        folds.append((
            df_target.iloc[i_train],
            df_target.iloc[i_test]
        ))
    
    args_list = [(name, fold_n, conf, params, model_type, train_target, valid_target, test_target)
                 for name, params in approaches_to_train.items()
                 for fold_n, (train_target, valid_target) in enumerate(folds)
                 #for model_type in ['xgb','lgb']
                 for model_type in ['fastai']
                 #for model_type in ['neural_automl']
                 ]

    if conf['pool']:
        pool = Pool(processes=conf['n_workers'])
        results = pool.map(train_and_score, args_list)
        df_results = pd.DataFrame(results).set_index('name')[['oof_accuracy', 'test_accuracy']]
    else:
        results = map(train_and_score, args_list)
        df_results = pd.DataFrame(results).set_index('name')[['oof_accuracy', 'test_accuracy']]

    # score already trained models on valid and tets sets
    args_list = [(name, conf, params, df_target, test_target) for name, params in approaches_to_score.items()]
    if conf['pool']:
        pool = Pool(processes=conf['n_workers'])
        results = reduce(iadd, pool.map(get_scores, args_list))
        df_scores = pd.DataFrame(results).set_index('name')[['oof_accuracy', 'test_accuracy']]
    else:
        results = reduce(iadd, map(get_scores, args_list))
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
