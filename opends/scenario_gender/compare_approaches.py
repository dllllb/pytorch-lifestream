import logging
import os
from multiprocessing import Pool

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from functools import reduce
from operator import iadd

from scenario_gender.features import load_features, load_scores
from scenario_gender.features import COL_ID, COL_TARGET
from dltranz.neural_automl.neural_automl_tools import train_from_config

logger = logging.getLogger(__name__)


def prepare_parser(parser):
    parser.add_argument('--n_workers', type=int, default=5)
    parser.add_argument('--cv_n_split', type=int, default=5)
    parser.add_argument('--data_path', type=os.path.abspath, default='../data/gender/')
    parser.add_argument('--test_size', type=float, default=0.4)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--model_seed', type=int, default=42)
    parser.add_argument('--ml_embedding_file_names', nargs='+', default=['embeddings.pickle'])
    parser.add_argument('--target_score_file_names', nargs='+', default=['target_scores', 'finetuning_scores'])
    parser.add_argument('--output_file', type=os.path.abspath, default='runs/scenario_gender.csv')
    parser.add_argument('--pos', type=int, nargs='*', default=[])


def read_target(conf):
    target = pd.read_csv(os.path.join(conf['data_path'], 'gender_train.csv'))
    test_ids = set(pd.read_csv(os.path.join(conf['data_path'], 'test_ids.csv'))[COL_ID].tolist())
    is_test = [(x in test_ids) for x in target[COL_ID]]

    logger.info(f'Train size: {len(target) - sum(is_test)} clients')
    logger.info(f'Test size: {sum(is_test)} clients')

    return target[[not x for x in is_test]].set_index(COL_ID), target[is_test].set_index(COL_ID)

def train_and_score(args):
    name, fold_n, conf, params, model_type, train_target, valid_target, test_target = args

    logger.info(f'[{name}:{model_type:6}:{fold_n}] Started: {params}')

    features = load_features(conf, **params)

    y_train = train_target[COL_TARGET]
    y_valid = valid_target[COL_TARGET]
    y_test = test_target[COL_TARGET]

    train_features = [df.reindex(index=train_target.index) for df in features]
    df_fn = [df.quantile(0.5) for df in train_features]
    df_norm = [df.max() - df.min() + 1e-5 for df in train_features]

    valid_features = [df.reindex(index=valid_target.index) for df in features]
    test_features = [df.reindex(index=test_target.index) for df in features]

    X_train = pd.concat([df.fillna(fn) / n for df, fn, n in zip(train_features, df_fn, df_norm)], axis=1)
    X_valid = pd.concat([df.fillna(fn) / n for df, fn, n in zip(valid_features, df_fn, df_norm)], axis=1)
    X_test = pd.concat([df.fillna(fn) / n for df, fn, n in zip(test_features, df_fn, df_norm)], axis=1)

    if model_type == 'linear':
        model = LogisticRegression()
    elif model_type == 'xgb':
        model = xgb.XGBClassifier(
            # objective='multi:softprob',
            # num_class=4,
            n_jobs=4,
            seed=conf['model_seed'],
            n_estimators=300)
    elif model_type == 'lgb':
        model = lgb.LGBMClassifier(
        n_estimators=500,
        boosting_type='gbdt',
        objective='binary',
        metric='auc',
        subsample=0.5,
        subsample_freq=1,
        learning_rate=0.02,
        feature_fraction=0.75,
        max_depth=6,
        lambda_l1=1,
        lambda_l2=1,
        min_data_in_leaf=50,
        random_state=conf['model_seed']
    )
    elif model_type == 'neural_automl':
        pass
    else:
        raise NotImplementedError(f'unknown model type {model_type}')

    if model_type != 'neural_automl':
        model.fit(X_train, y_train)
        valid_rocauc_score = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
        test_rocauc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        valid_rocauc_score = train_from_config(X_train.values, 
                                               y_train.values.astype('float32'), 
                                               X_valid.values, 
                                               y_valid.values.astype('float32'),
                                               'gender.json')
        test_rocauc_score('not supported yet')

    logger.info(
        ' '.join([
            f'[{name:10}:{model_type:6}:{fold_n}]',
            'Finished with rocauc_score',
            f'valid={valid_rocauc_score:.4f},',
            f'test={test_rocauc_score:.4f}',
            f': {params}'
        ]))

    res = {}
    res['name'] = '_'.join([model_type, name])
    res['model_type'] = model_type
    res['fold_n'] = fold_n
    res['oof_rocauc_score'] = valid_rocauc_score
    res['test_rocauc_score'] = test_rocauc_score
    return res


def get_scores(args):
    name, conf, params, df_target, test_target = args

    logger.info(f'[{name}] Scoring started: {params}')

    result = []
    valid_scores, test_scores = load_scores(conf, **params)
    for fold_n, (valid_fold, test_fold) in enumerate(zip(valid_scores, test_scores)):

        valid_fold = valid_fold.merge(df_target, on=COL_ID, how = 'left')
        test_fold = test_fold.merge(test_target, on=COL_ID, how = 'left')

        result.append({
            'name' : name,
            'fold_n' : fold_n,
            'oof_rocauc_score' : roc_auc_score(valid_fold[COL_TARGET], valid_fold.iloc[:,0]),
            'test_rocauc_score' : roc_auc_score(test_fold[COL_TARGET], test_fold.iloc[:,0])
        })

    return result


def main(conf):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    approaches_to_train = {
        'baseline' : {'use_client_agg': True, 'use_mcc_code_stat': True, 'use_tr_type_stat': True},
        **{
            f"embeds: {file_name}" : {'metric_learning_embedding_name': file_name} for file_name in conf['ml_embedding_file_names']
        }
    }

    approaches_to_score = {
        f"scores: {file_name}" : {'target_scores_name': file_name} for file_name in conf['target_score_file_names']
    }

    df_target, test_target = read_target(conf)
    
    # train model on features and score valid and test sets
    folds = []
    skf = StratifiedKFold(n_splits=conf['cv_n_split'], random_state=conf['random_state'], shuffle=True)
    for i_train, i_test in skf.split(df_target, df_target[COL_TARGET]):
        folds.append((
            df_target.iloc[i_train],
            df_target.iloc[i_test]
        ))

    args_list = [(name, fold_n, conf, params, model_type, train_target, valid_target, test_target)
                 for name, params in approaches_to_train.items()
                 for fold_n, (train_target, valid_target) in enumerate(folds)
                 for model_type in ['xgb', 'linear','lgb']
                 ]

    pool = Pool(processes=conf['n_workers'])
    results = pool.map(train_and_score, args_list)
    df_results = pd.DataFrame(results).set_index('name')[['oof_rocauc_score','test_rocauc_score']]

    # score already trained models on valid and tets sets
    pool = Pool(processes=conf['n_workers'])
    args_list = [(name, conf, params, df_target, test_target) for name, params in approaches_to_score.items()]
    results = reduce(iadd, pool.map(get_scores, args_list))
    df_scores = pd.DataFrame(results).set_index('name')[['oof_rocauc_score','test_rocauc_score']]

    # combine results
    df_results = pd.concat([df_results, df_scores])
    df_results = pd.concat([
        df_results.groupby(level='name')[['oof_rocauc_score']].agg([
            'mean', 'std', lambda x: '[' + ' '.join([f'{i:.3f}' for i in sorted(x)]) + ']']),
        df_results.groupby(level='name')[['test_rocauc_score']].agg([
            'mean', 'std', lambda x: '[' + ' '.join([f'{i:.3f}' for i in sorted(x)]) + ']']),
    ], axis=1).sort_index()

    with pd.option_context(
            'display.float_format', '{:.4f}'.format,
            'display.max_columns', None,
            'display.expand_frame_repr', False,
    ):
        logger.info(f'Results:\n{df_results}')
        with open(conf['output_file'], 'w') as f:
            print(df_results, file=f)
