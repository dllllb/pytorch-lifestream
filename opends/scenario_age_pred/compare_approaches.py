import logging
import os
from multiprocessing import Pool

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from scenario_age_pred.features import load_features
from dltranz.neural_automl.neural_automl_tools import train_from_config

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


def read_target(conf):
    target = pd.read_csv(os.path.join(conf['data_path'], 'train_target.csv')).set_index('client_id')
    return target


def get_scores(args):
    pos, fold_n, conf, params, model_type, train_target, valid_target = args

    logger.info(f'[{pos:4}:{fold_n}] Started: {params}')

    features = load_features(conf, **params)

    y_train = train_target['bins']
    y_valid = valid_target['bins']

    X_train = pd.concat([df.reindex(index=train_target.index) for df in features], axis=1)
    X_valid = pd.concat([df.reindex(index=valid_target.index) for df in features], axis=1)

    if model_type == 'xgb':
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            n_jobs=4,
            seed=conf['model_seed'],
            n_estimators=300)
    elif model_type == 'neural_automl':
        pass
    else:
        raise NotImplementedError(f'unknown model type {model_type}')

    if model_type != 'neural_automl':
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        accuracy = (y_valid == pred).mean()
    else:
        accuracy = train_from_config(X_train.values, 
                                     y_train.values.astype('long'), 
                                     X_valid.values, 
                                     y_valid.values.astype('long'),
                                     'age.json')

    logger.info(f'[{pos:4}:{fold_n}] Finished with accuracy {accuracy:.4f}: {params}')

    res = params.copy()
    res['pos'] = pos
    res['fold_n'] = fold_n
    res['accuracy'] = accuracy
    return res


def main(conf):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    param_list = [
        {'use_random': True},
        {'use_client_agg': True},
        {'use_small_group_stat': True},
        {'use_client_agg': True, 'use_small_group_stat': True},
    ] + [
        {'metric_learning_embedding_name': file_name} for file_name in conf['ml_embedding_file_names']
    ] + [
        {'target_scores_name': file_name} for file_name in conf['target_score_file_names']
    ]

    df_target = read_target(conf)
    df_target = df_target
    folds = []
    skf = StratifiedKFold(n_splits=conf['cv_n_split'], random_state=conf['random_state'], shuffle=True)
    for i_train, i_test in skf.split(df_target, df_target['bins']):
        folds.append((
            df_target.iloc[i_train],
            df_target.iloc[i_test]
        ))

    args_list = [(pos, fold_n, conf, params, model_type, train_target, valid_target)
                 for pos, params in enumerate(param_list)
                 for fold_n, (train_target, valid_target) in enumerate(folds)
                 for model_type in ['xgb', 'neural_automl']
                 ]

    pool = Pool(processes=conf['n_workers'])
    results = pool.map(get_scores, args_list)
    df_results = pd.DataFrame(results).set_index('pos').drop(columns='fold_n')
    df_results = pd.concat([
        df_results.groupby(level='pos')[['accuracy']].agg([
            'mean', 'std', lambda x: '[' + ' '.join([f'{i:.3f}' for i in sorted(x)]) + ']']),
        df_results.drop(columns='accuracy').groupby(level='pos').first(),
    ], axis=1).sort_index()

    with pd.option_context(
        'display.float_format', '{:.4f}'.format,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
    ):
        logger.info(f'Results:\n{df_results}')
        with open(conf['output_file'], 'w') as f:
            print(df_results, file=f)
