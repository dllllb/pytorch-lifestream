import logging
from functools import partial, reduce
from operator import iadd

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score
from scipy.special import softmax
from math import sqrt

import dltranz.scenario_cls_tools as sct
from dltranz.metric_learn.metric import PercentR2Metric, PercentPredictMetric
from scenario_spend_only_rur.const import (
    DEFAULT_DATA_PATH, DEFAULT_RESULT_FILE, TEST_IDS_FILE, DATASET_FILE, COL_ID, COL_TARGET,
)
from scenario_spend_only_rur.features import load_scores

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

        def numpy_estim(y_pred, y):
          soft_pred = softmax(y_pred, axis=1)
          rss = np.linalg.norm(soft_pred - y, ord=1, axis=1)**2
          apriori_mean = y.mean(axis=0)
          apriori_mean = np.repeat(np.expand_dims(apriori_mean,0), y_pred.shape[0], 0)
          tss = np.linalg.norm(soft_pred - apriori_mean, ord=1, axis=1)**2
          r2 = 1 - rss/tss
          return np.mean(r2).item()
        
        y_pred = valid_fold.iloc[:,1:53].to_numpy()
        y = valid_fold.iloc[:,55:].to_numpy() 
        valid_metr = numpy_estim(y_pred, y)
        y_pred = test_fold.iloc[:,1:53].to_numpy()
        y = test_fold.iloc[:,55:].to_numpy() 
        test_metr = numpy_estim(y_pred, y)
        result.append({
            'name': name,
            'fold_n': fold_n,
            'oof_accuracy': valid_metr,
            'test_accuracy': test_metr
        })

    return result


def main(conf):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    approaches_to_score = {
        f"scores: {file_name}": {'target_scores_name': file_name}
        for file_name in conf['score_file_names']
    }

    pool = sct.WPool(processes=conf['n_workers'])
    df_results = None
    df_scores = None

    df_target, test_target = sct.read_train_test(conf['data_path'], DATASET_FILE, TEST_IDS_FILE, COL_ID)

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
