import logging
import os
from functools import partial
from multiprocessing import Pool

import pandas as pd
from sklearn.metrics import make_scorer, balanced_accuracy_score

from dltranz.scenario_cls_tools import prepare_common_parser, read_train_test, get_folds, drop_na_target, \
    train_and_score, KWParamsTrainAndScore, filter_infrequent_target, group_stat_results
from scenario_tin_cls.const import (
    DEFAULT_DATA_PATH, DEFAULT_RESULT_FILE, DATASET_FILE, TEST_IDS_FILE, COL_ID, COL_TARGET,
)
from scenario_tin_cls.features import load_features

logger = logging.getLogger(__name__)

MODEL_TYPES = ['linear', 'xgb']


def prepare_parser(parser):
    return prepare_common_parser(parser, data_path=DEFAULT_DATA_PATH, output_file=DEFAULT_RESULT_FILE)


def main(conf):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    pool = Pool(processes=conf['n_workers'])

    for col_i, col_target in enumerate(COL_TARGET):
        logger.info(f'=== Start for col_target={col_target} [{col_i + 1:2} / {len(COL_TARGET)}] ===')

        approaches_to_train = {
            'random': {'use_random': True},
            'baseline trx': {'use_trans_common_features': True, 'use_trans_mcc_features': True},
            'trans_common_features': {'use_trans_common_features': True},
            'trans_mcc_features': {'use_trans_mcc_features': True},
            **{
                f"embeds: {file_name}": {'metric_learning_embedding_name': file_name}
                for file_name in conf['ml_embedding_file_names']
            }
        }

        approaches_to_score = {
        }

        df_target, test_target = read_train_test(conf['data_path'], DATASET_FILE, TEST_IDS_FILE, COL_ID)
        df_target = drop_na_target(col_target, df_target, 'train')
        test_target = drop_na_target(col_target, test_target, 'test')

        target_value_counts = df_target[col_target].value_counts(normalize=True)
        logger.info(f'target_value_counts:\n{target_value_counts}')

        target_values = target_value_counts[lambda x: x > 0.05].index.tolist()
        df_target = filter_infrequent_target(col_target, df_target, 'train', target_values)
        test_target = filter_infrequent_target(col_target, test_target, 'test', target_values)
        target_value_counts = df_target[col_target].value_counts(normalize=True)
        logger.info(f'target_value_counts:\n{target_value_counts}')

        folds = get_folds(df_target, col_target, conf['cv_n_split'], conf['random_state'])

        # train-test on features
        scorer_name = 'balanced_accuracy_score'
        args_list = [
            KWParamsTrainAndScore(
                name=name,
                fold_n=fold_n,
                load_features_f=partial(load_features, conf, **params),
                model_type=model_type,
                model_seed=conf['model_seed'],
                scorer_name=scorer_name,
                scorer=make_scorer(balanced_accuracy_score),
                col_target=col_target,
                df_train=train_target,
                df_valid=valid_target,
                df_test=test_target,
            )
            for fold_n, (train_target, valid_target) in enumerate(folds)
            for model_type in MODEL_TYPES
            for name, params in approaches_to_train.items()
        ]

        results = pool.map(train_and_score, args_list)
        df_results = pd.DataFrame(results).assign(name=lambda x: col_target + ': ' + x['name']).set_index('name')

        df_results = group_stat_results(df_results, 'name', [f'oof_{scorer_name}', f'test_{scorer_name}'])
        with pd.option_context(
                'display.float_format', '{:.4f}'.format,
                'display.max_columns', None,
                'display.expand_frame_repr', False,
                'display.max_colwidth', 100,
        ):
            logger.info(f'Results:\n{df_results}')
            output_file_name = os.path.splitext(conf['output_file'])
            output_file_name = ''.join([output_file_name[0], '_', col_target, output_file_name[1]])

            logger.info(f'Saved to "{output_file_name}"')
            with open(output_file_name, 'w') as f:
                print(df_results, file=f)

        logger.info(f'=== Finish for col_target={col_target} [{col_i + 1:2} / {len(COL_TARGET)}] ===')
