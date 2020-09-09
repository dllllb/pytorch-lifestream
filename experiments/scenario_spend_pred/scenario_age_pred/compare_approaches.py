import logging
from functools import partial, reduce
from operator import iadd

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score

import dltranz.scenario_cls_tools as sct
from scenario_age_pred.const import (
    DEFAULT_DATA_PATH, DEFAULT_RESULT_FILE, TEST_IDS_FILE, DATASET_FILE, COL_ID, COL_TARGET,
)
from scenario_age_pred.features import load_features, load_scores

logger = logging.getLogger(__name__)


def prepare_parser(parser):
    sct.prepare_common_parser(parser, data_path=DEFAULT_DATA_PATH, output_file=DEFAULT_RESULT_FILE)


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

    embedding_file_names = sct.expand_path(conf['data_path'], conf['embedding_file_names'])
    approaches_to_train = {
        **{
            f"embeds: {file_name}": {'metric_learning_embedding_name': file_name}
            for file_name in embedding_file_names
        },
    }
    if conf['add_baselines']:
        approaches_to_train.update({
            'baseline': {'use_client_agg': True, 'use_small_group_stat': True},
        })

    if conf['add_baselines'] and conf['add_emb_baselines']:
        approaches_to_train.update({
            f"embeds: {file_name} and baseline": {
                'metric_learning_embedding_name': file_name, 'use_client_agg': True, 'use_small_group_stat': True}
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
    if len(approaches_to_train) > 0:
        folds = sct.get_folds(df_target, COL_TARGET, conf['cv_n_split'], conf['random_state'], conf.get('labeled_amount',-1))

        model_types = {
            'xgb': dict(
                objective='multi:softprob',
                num_class=4,
                n_jobs=4,
                seed=conf['model_seed'],
                n_estimators=300,
            ),
            'linear': dict(),
            'lgb': dict(
                n_estimators=1000,
                boosting_type='gbdt',
                objective='multiclass',
                num_class=4,
                metric='multi_error',
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
                n_jobs=4,
            ),
        }

        # train and score models
        args_list = [sct.KWParamsTrainAndScore(
            name=name,
            fold_n=fold_n,
            load_features_f=partial(load_features, conf=conf, **params),
            model_type=model_type,
            model_params=model_params,
            scorer_name='accuracy',
            scorer=make_scorer(accuracy_score),
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
        df_results = pd.DataFrame(results).set_index('name')[['oof_accuracy', 'test_accuracy']]
    if len(approaches_to_score) > 0:
        # score already trained models on valid and test sets
        args_list = [(name, conf, params, df_target, test_target) for name, params in approaches_to_score.items()]
        results = reduce(iadd, pool.map(get_scores, args_list))
        df_scores = pd.DataFrame(results).set_index('name')[['oof_accuracy', 'test_accuracy']]

    # combine results
    df_results = pd.concat([df for df in [df_results, df_scores] if df is not None])
    df_results = sct.group_stat_results(df_results, 'name', ['oof_accuracy', 'test_accuracy'])

    with pd.option_context(
            'display.float_format', '{:.4f}'.format,
            'display.max_columns', None,
            'display.max_rows', None,
            'display.expand_frame_repr', False,
            'display.max_colwidth', 100,
    ):
        logger.info(f'Results:\n{df_results}')
        with open(conf['output_file'], 'w') as f:
            print(df_results, file=f)
