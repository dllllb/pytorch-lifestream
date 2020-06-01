import logging
import os
from functools import partial, reduce
from operator import iadd

import pandas as pd
from glob import glob
from sklearn.metrics import roc_auc_score, make_scorer

import dltranz.scenario_cls_tools as sct
from scenario_x5.const import (
    COL_ID, COL_TARGET, DEFAULT_DATA_PATH, DEFAULT_RESULT_FILE, DATASET_FILE, TEST_IDS_FILE,
)
from scenario_x5.features import load_features, load_scores

logger = logging.getLogger(__name__)


def prepare_parser(parser):
    sct.prepare_common_parser(parser, data_path=DEFAULT_DATA_PATH, output_file=DEFAULT_RESULT_FILE)


def filter_target(df, col_target_name):
    mapping = {
        'F': 0,
        'M': 1,
    }

    if col_target_name == 'gender':
        return df[lambda x: x[col_target_name].isin(mapping.keys())]
    else:
        raise AttributeError(f'Unknown col_target_name: {col_target_name}')


def get_scores(args):
    name, conf, params, df_target, test_target = args

    logger.info(f'[{name}] Scoring started: {params}')

    result = []
    valid_scores, test_scores = load_scores(conf, **params)
    for fold_n, (valid_fold, test_fold) in enumerate(zip(valid_scores, test_scores)):
        valid_fold = valid_fold.merge(df_target, on=COL_ID, how='left')
        test_fold = test_fold.merge(test_target, on=COL_ID, how='left')

        result.append({
            'name': name,
            'fold_n': fold_n,
            'oof_rocauc_score': roc_auc_score(valid_fold[COL_TARGET], valid_fold.iloc[:, 0]),
            'test_rocauc_score': roc_auc_score(test_fold[COL_TARGET], test_fold.iloc[:, 0])
        })

    return result


def expand_path(data_path, wc_paths):
    data_path = os.path.join(data_path, '')  # ensure `/` at the end

    embedding_file_names = []
    for path in wc_paths:
        for n_path in glob(data_path + path):
            embedding_file_names.append(n_path[len(data_path):])
    logger.info(f'Found {len(embedding_file_names)} embedding files: [{embedding_file_names}]')
    return embedding_file_names


def main(conf):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    embedding_file_names = expand_path(conf['data_path'], conf['embedding_file_names'])
    approaches_to_train = {
        **{
            f"embeds: {file_name}": {'metric_learning_embedding_name': file_name}
            for file_name in embedding_file_names
        },
    }
    if conf['baseline_name']:
        approaches_to_train.update({
            'baseline': {'metric_learning_embedding_name': conf['baseline_name']},
        })
        approaches_to_train.update({
            f"embeds: {file_name} and baseline": {
                'metric_learning_embedding_name': [file_name, conf['baseline_name']]
            }
            for file_name in embedding_file_names
        })

    approaches_to_score = {
        f"scores: {file_name}": {'target_scores_name': file_name}
        for file_name in conf['score_file_names']
    }

    pool = sct.WPool(processes=conf['n_workers'])
    df_results = None
    df_scores = None

    df_target, test_target = sct.read_train_test(conf['data_path'], DATASET_FILE, TEST_IDS_FILE, COL_ID)
    df_target = filter_target(df_target, COL_TARGET)
    test_target = filter_target(test_target, COL_TARGET)
    if len(approaches_to_train) > 0:
        logger.info(f'Found {len(approaches_to_train)} options for `train_and_score`')
        folds = sct.get_folds(df_target, COL_TARGET, conf['cv_n_split'], conf['random_state'], conf.get('labeled_amount',-1))

        model_types = {
            'xgb': dict(
                n_jobs=4,
                seed=conf['model_seed'],
                n_estimators=300,
            ),
            'linear': dict(),
            'lgb': dict(
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
                random_state=conf['model_seed'],
                n_jobs=8,
            ),
        }

        # train and score models
        args_list = [sct.KWParamsTrainAndScore(
            name=name,
            fold_n=fold_n,
            load_features_f=partial(load_features, conf=conf, **params),
            model_type=model_type,
            model_params=model_params,
            scorer_name='rocauc_score',
            scorer=make_scorer(roc_auc_score, needs_proba=True),
            col_target=COL_TARGET,
            df_train=train_target,
            df_valid=valid_target,
            df_test=test_target,
        )
            for name, params in approaches_to_train.items()
            for fold_n, (train_target, valid_target) in enumerate(folds)
            for model_type, model_params in model_types.items() if model_type in conf['models']
        ]
        results = []
        for i, r in enumerate(pool.imap_unordered(sct.train_and_score, args_list)):
            results.append(r)
            logger.info(f'Done {i + 1:4d} from {len(args_list)}')
        df_results = pd.DataFrame(results).set_index('name')[['oof_rocauc_score', 'test_rocauc_score']]

    if len(approaches_to_score) > 0:
        # score already trained models on valid and test sets
        args_list = [(name, conf, params, df_target, test_target) for name, params in approaches_to_score.items()]
        results = reduce(iadd, pool.map(get_scores, args_list))
        df_scores = pd.DataFrame(results).set_index('name')[['oof_rocauc_score', 'test_rocauc_score']]

    # combine results
    df_results = pd.concat([df for df in [df_results, df_scores] if df is not None])
    df_results = sct.group_stat_results(df_results, 'name', ['oof_rocauc_score', 'test_rocauc_score'])

    with pd.option_context(
            'display.float_format', '{:.4f}'.format,
            'display.max_columns', None,
            'display.max_rows', None,
            'display.expand_frame_repr', False,
            'display.max_colwidth', 200,
    ):
        logger.info(f'Results:\n{df_results}')
        with open(conf['output_file'], 'w') as f:
            print(df_results, file=f)
