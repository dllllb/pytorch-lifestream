import json
import logging
import os

import pandas as pd
from pandas.io.json import json_normalize

from dltranz.util import group_stat_results


logger = logging.getLogger(__name__)


def save_result(config, fold_n, score, metrics):
    history_file = config.history_file

    if os.path.isfile(history_file) and os.stat(history_file).st_size > 0:
        with open(history_file, 'rt') as f:
            history = json.load(f)
    else:
        history = []

    history.append({
        'name': config.name,
        'fold_n': fold_n,
        'config': vars(config),
        'final_score': score,
        'metrics': metrics,
    })

    with open(history_file, 'wt') as f:
        json.dump(history, f, indent=2)


def prepare_parser(sub_parser):
    sub_parser.add_argument('--report_file', type=os.path.abspath, required=False)


def main(config):
    history_file = config.history_file
    report_file = config.report_file

    with open(history_file, 'rt') as f:
        history = json.load(f)

    df = json_normalize(history)

    metrics_prefix = 'final_score.'
    metric_columns = [col for col in df.columns if col.startswith(metrics_prefix)]
    old_metric = metric_columns
    metric_columns = [col[len(metrics_prefix):] for col in old_metric]
    df_results = df[['name', 'fold_n'] + old_metric].rename(
        columns={k: v for k, v in zip(old_metric, metric_columns)})

    df_results = group_stat_results(df_results, 'name', metric_columns, ['fold_n'])

    with pd.option_context(
        'display.float_format', '{:.4f}'.format,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
    ):
        logger.info(f'Results:\n{df_results}')
        with open(report_file, 'w') as f:
            print(df_results, file=f)
