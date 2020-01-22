import logging
import os
from multiprocessing import Pool

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from scenario_age_pred.features import load_features

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
    target = pd.read_csv(os.path.join(conf['data_path'], 'train_target.csv'))
    test_ids = set(pd.read_csv(os.path.join(conf['data_path'], 'test_ids.csv'))['client_id'].tolist())
    is_test = [(x in test_ids) for x in target['client_id']]

    logger.info(f'Train size: {len(target) - sum(is_test)} clients')
    logger.info(f'Test size: {sum(is_test)} clients')

    return target[[not x for x in is_test]].set_index('client_id'), target[is_test].set_index('client_id')


def get_scores(args):
    name, fold_n, conf, params, train_target, valid_target, test_target = args

    logger.info(f'[{name}:{fold_n}] Started: {params}')

    features = load_features(conf, **params)

    y_train = train_target['bins']
    y_valid = valid_target['bins']
    y_test = test_target['bins']

    X_train = pd.concat([df.reindex(index=train_target.index) for df in features], axis=1)
    X_valid = pd.concat([df.reindex(index=valid_target.index) for df in features], axis=1)
    X_test = pd.concat([df.reindex(index=test_target.index) for df in features], axis=1)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        n_jobs=4,
        seed=conf['model_seed'],
        n_estimators=1) # !!! debug. 

    model.fit(X_train, y_train)
    valid_accuracy = (y_valid == model.predict(X_valid)).mean()
    test_accuracy = (y_test == model.predict(X_test)).mean()

    logger.info(f'[{name:4}:{fold_n}] Finished with accuracy valid={valid_accuracy:.4f}, test={test_accuracy:.4f}: {params}')

    res = params.copy()
    res['name'] = name
    res['fold_n'] = fold_n
    res['oof_accuracy'] = valid_accuracy
    res['test_accuracy'] = test_accuracy
    return res


def main(conf):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    approaches_to_train = {
        'baseline' : {'use_client_agg': True, 'use_small_group_stat': True},
    }
    approaches_to_train = {
        **approaches_to_train,
        **{
            f"embeds: {file_name}" : {'metric_learning_embedding_name': file_name} for file_name in conf['ml_embedding_file_names']
        },
        **{
        f"scores: {file_name}" : {'target_scores_name': file_name} for file_name in conf['target_score_file_names']
        }
    }

    df_target, test_target  = read_target(conf)

    folds = []
    skf = StratifiedKFold(n_splits=conf['cv_n_split'], random_state=conf['random_state'], shuffle=True)
    for i_train, i_test in skf.split(df_target, df_target['bins']):
        folds.append((
            df_target.iloc[i_train],
            df_target.iloc[i_test]
        ))

    args_list = [(name, fold_n, conf, params, train_target, valid_target, test_target)
                 for name, params in approaches_to_train.items()
                 for fold_n, (train_target, valid_target) in enumerate(folds)]

    pool = Pool(processes=conf['n_workers'])
    results = pool.map(get_scores, args_list)
    df_results = pd.DataFrame(results).set_index('name')[['oof_accuracy','test_accuracy']]
    df_results = pd.concat([
        df_results.groupby(level='name')[['oof_accuracy']].agg([
            'mean', 'std', lambda x: '[' + ' '.join([f'{i:.3f}' for i in sorted(x)]) + ']']),
        df_results.drop(columns='oof_accuracy').groupby(level='name').first(),
        df_results.groupby(level='name')[['test_accuracy']].agg([
            'mean', 'std', lambda x: '[' + ' '.join([f'{i:.3f}' for i in sorted(x)]) + ']']),
        df_results.drop(columns='test_accuracy').groupby(level='name').first(),
    ], axis=1).sort_index()

    with pd.option_context(
        'display.float_format', '{:.4f}'.format,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
    ):
        logger.info(f'Results:\n{df_results}')
        with open(conf['output_file'], 'w') as f:
            print(df_results, file=f)
