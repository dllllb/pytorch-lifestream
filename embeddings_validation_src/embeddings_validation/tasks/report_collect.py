import json
import os
import datetime

import numpy as np
import pandas as pd
from pandas import json_normalize
import scipy.stats

import luigi

from embeddings_validation.config import Config
from embeddings_validation.tasks.external_score import ExternalScore
from embeddings_validation.tasks.fold_estimator import FoldEstimator


def print_float_list(x, float_format):
    x = x.tolist()
    return '[' + ' '.join([float_format.format(i) for i in x]) + ']'


def print_str_list(x):
    x = x.tolist()
    return '[' + ' '.join(x) + ']'


def t_interval(x, p=0.95):
    eps = 1e-9
    n = len(x)
    s = x.std(ddof=1)

    return scipy.stats.t.interval(p, n - 1, loc=x.mean(), scale=(s + eps) / (n ** 0.5))


def t_int_l(x, p=0.95):
    return t_interval(x, p)[0]


def t_int_h(x, p=0.95):
    return t_interval(x, p)[1]


def t_pm(x, p=0.95):
    l, h = t_interval(x, p)
    return (h - l) / 2


def fisher_var_test(x, y):
    """
    http://www.machinelearning.ru/wiki/index.php?title=Критерий_Фишера

    H[0]: s_1^2 == s_2^2
    H[1]: s_1^2 != s_2^2

    return True if H[0] should be rejected
    """
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    n1, n2 = len(x), len(y)

    f = v1 / v2
    sf = scipy.stats.f(n1 - 1, n2 - 1)
    return f, sf


def t_test(x, y):
    """
    http://www.machinelearning.ru/wiki/index.php?title=Критерий_Стьюдента

    H[0]: m1 == m2
    H[1]: m1 < m2

    return t-stat
    """
    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    n1, n2 = len(x), len(y)

    t = m2 - m1
    t = t / (((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) ** 0.5
    t = t * ((n1 * n2) / (n1 + n2)) ** 0.5
    return t, scipy.stats.t(n1 + n2 - 2)


def t_test_rev(x, y, alpha=0.05):
    """
    http://www.machinelearning.ru/wiki/index.php?title=Критерий_ Стьюдента

    H[0]: m1 == m2 - A
    H[1]: m1 < m2 - A

    return A
    """
    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    n1, n2 = len(x), len(y)

    t = scipy.stats.t(n1 + n2 - 2).ppf(1 - alpha)
    t = t / ((n1 * n2) / (n1 + n2)) ** 0.5
    t = t * (((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) ** 0.5
    return m2 - m1 - t


def t_test_delta_intervals(x, y, alpha=0.05):
    """
    http://www.machinelearning.ru/wiki/index.php?title=Критерий_ Стьюдента

    delta = m2 - m1

    return delta, delta_pm
    """
    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    n1, n2 = len(x), len(y)

    delta = m2 - m1

    t = scipy.stats.t(n1 + n2 - 2).ppf(1 - alpha / 2)
    t = t / ((n1 * n2) / (n1 + n2)) ** 0.5
    t = t * (((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) ** 0.5

    return delta, t


def get_delta_a_value(col, scores_x, baseline):
    """

    :param col:
    :param scores_x:
    :param baseline:
    :return: [t_f_stat, t_f_alpha, t_t_stat, t_t_alpha, t_A, t_A_pp]
        where
        t_A - is absolute delta, which can be added to `baseline`,
            and `scores_x` still significant bigger than `baseline`
    """
    scores_x = np.array(scores_x)
    baseline = baseline.values
    result_fields = [
        (col, 't_f_stat'),
        (col, 't_f_alpha'),
        (col, 't_t_stat'),
        (col, 't_t_alpha'),
        (col, 't_A'),
        (col, 't_A_pp'),
    ]

    f, sf = fisher_var_test(baseline, scores_x)
    t, st = t_test(baseline, scores_x)
    A = t_test_rev(baseline, scores_x)

    return pd.Series(data=[
        f,
        sf.cdf(f),
        t,
        st.cdf(t),
        A,
        A / baseline.mean() * 100,
    ], index=result_fields)


def get_delta_intervals(col, scores_x, baseline):
    """

    :param col:
    :param scores_x:
    :param baseline:
    :return: [t_f_stat, t_f_alpha, t_t_stat, t_t_alpha,
              t_delta, t_delta_pm, t_delta_l, t_delta_h,
              t_delta_pp, t_delta_pm_pp, t_delta_l_pp, t_delta_h_pp]
        where
        t_delta - show how `scores_x` still bigger than `baseline`
        t_delta_pm - is confidence interval delta for t_delta
        t_delta_l = t_delta - t_delta_pm
        t_delta_h = t_delta + t_delta_pm
    """
    scores_x = np.array(scores_x)
    baseline = baseline.values
    result_fields = [
        (col, 't_f_stat'),
        (col, 't_f_alpha'),
        (col, 't_t_stat'),
        (col, 't_t_alpha'),
        #
        (col, 't_delta'),
        (col, 't_delta_pm'),
        (col, 't_delta_l'),
        (col, 't_delta_h'),
        #
        (col, 't_delta_pp'),
        (col, 't_delta_pm_pp'),
        (col, 't_delta_l_pp'),
        (col, 't_delta_h_pp'),
    ]

    f, sf = fisher_var_test(baseline, scores_x)
    t, st = t_test(baseline, scores_x)
    delta, delta_pm = t_test_delta_intervals(baseline, scores_x)

    return pd.Series(data=[
        f,
        sf.cdf(f),
        t,
        st.cdf(t),
        #
        delta,
        delta_pm,
        delta - delta_pm,
        delta + delta_pm,
        #
        delta / baseline.mean() * 100,
        delta_pm / baseline.mean() * 100,
        (delta - delta_pm) / baseline.mean() * 100,
        (delta + delta_pm) / baseline.mean() * 100,
    ], index=result_fields)


class ReportByFolds(luigi.Task):
    conf = luigi.Parameter()
    total_cpu_count = luigi.IntParameter()

    model_name = luigi.Parameter()
    feature_name = luigi.Parameter()

    def requires(self):
        conf = Config.read_file(self.conf)

        for fold_id in conf.folds:
            yield FoldEstimator(
                conf=self.conf,
                model_name=self.model_name,
                feature_name=self.feature_name,
                fold_id=fold_id,
                total_cpu_count=self.total_cpu_count,
            )

    def output(self):
        for target in self.input():
            yield target


class ReportCollect(luigi.Task):
    conf = luigi.Parameter()
    total_cpu_count = luigi.IntParameter()

    f = None

    def requires(self):
        conf = Config.read_file(self.conf)

        for model_name in conf.models:
            for feature_name in conf.features:
                yield ReportByFolds(
                    conf=self.conf,
                    model_name=model_name,
                    feature_name=feature_name,
                    total_cpu_count=self.total_cpu_count,
                )
        for name, external_path in conf.external_scores.items():
            yield ExternalScore(
                conf=self.conf,
                name=name,
                external_path=external_path,
            )

    def output(self):
        conf = Config.read_file(self.conf)

        path = os.path.join(conf.root_path, conf['report_file'])
        return luigi.LocalTarget(path)

    def load_results(self):
        parts = []
        total_count = 0
        error_count = 0
        for i in self.input():
            if isinstance(i, luigi.LocalTarget):
                i = [i]
            for i2 in i:
                total_count += 1
                with open(i2.path, 'r') as f:
                    scores = json.load(f)

                if len(scores) == 0:
                    error_count += 1
                    os.remove(i2.path)
                parts.extend(scores)

        if len(parts) == 0:
            raise AttributeError('No parts are found')

        pd_report = json_normalize(parts, max_level=1)
        pd_report = pd_report.melt(id_vars=['model_name', 'feature_name', 'fold_id'], var_name='_metric')
        pd_report = pd.concat([
            pd_report,
            pd_report['_metric'].str.extract(r'(?P<split_name>\w+)\.(?P<metric_name>[\w\.]+)'),
        ], axis=1)
        pd_report = pd_report.set_index(['split_name', 'metric_name', 'model_name', 'feature_name', 'fold_id'])['value']
        pd_report = pd_report.sort_index()
        return pd_report, total_count, error_count

    def run(self):
        conf = Config.read_file(self.conf)

        splits = []
        if conf['report.is_check_train']:
            splits.append('scores_train')
        splits.append('scores_valid')
        if 'test_id' in conf['split']:
            splits.append('scores_test')

        pd_report, total_count, error_count = self.load_results()
        pd_split_report = pd_report.loc[splits].unstack(0).reindex(columns=splits)
        metric_index = {m: pd_split_report for m in pd_split_report.index.get_level_values(0).unique()}
        if 'process_info' in pd_report:
            pd_process_report = pd_report.loc[['process_info']].unstack(0)
            metric_index.update({m: pd_process_report for m in pd_process_report.index.get_level_values(0).unique()})

        with self.output().open('w') as f:
            self.f = f

            self.print_header()
            self.print_errors(total_count, error_count)

            for k in conf['report.metrics'].keys():
                self.print_row_pandas(k, metric_index[k].loc[k], **conf['report.metrics'].get(k, {}))
                del metric_index[k]

            if conf['report.print_all_metrics'] and len(metric_index) > 0:
                print('Other metrics:', file=f)
                for k in metric_index:
                    self.print_row_pandas(k, metric_index[k].loc[k])

            self.print_footer()

    def print_header(self):
        self.print_line()
        _text = f"""Vector testing report
Params:
    conf: "{self.conf}"
"""
        print(_text, file=self.f)

    def print_errors(self, total_count, error_count):
        print(f"Collected {total_count} files with {error_count} errors", file=self.f)
        if error_count > 0:
            print(f"Check logs for detail information", file=self.f)
        print('', file=self.f)

    def print_row_pandas(self, metric_name, df_row,
                         keep_columns=None, float_format='{:.4f}',
                         baseline_key=None, split_columns=False):
        self.print_line()
        with pd.option_context(
                'display.float_format', float_format.format,
                'display.max_columns', None,
                'display.max_rows', None,
                'display.expand_frame_repr', False,
                'display.max_colwidth', None,
        ):
            def values(x):
                if is_numeric:
                    return print_float_list(x, float_format=float_format)
                else:
                    return print_str_list(x)

            m_list = {
                'mean': 'mean',
                't_pm': t_pm,
                't_int_l': t_int_l,
                't_int_h': t_int_h,
                'std': 'std',
                'values': values,
                'first': 'first'
            }
            m_default_numeric_list = ['mean', 't_pm', 't_int_l', 't_int_h', 'std', 'values']
            m_default_str_list = ['first', 'values']

            print(f'Metric: "{metric_name}"', file=self.f)
            df = df_row.copy()
            main_columns = df.columns

            try:
                for col in df_row.columns:
                    df[col] = pd.to_numeric(df_row[col])
                is_numeric = True
            except Exception:
                df = df_row.astype(str)
                is_numeric = False

            if keep_columns is None:
                if is_numeric:
                    keep_columns = m_default_numeric_list
                else:
                    keep_columns = m_default_str_list

            metrics = [m_list[i] for i in keep_columns]

            df = df.groupby(['model_name', 'feature_name']).agg(metrics)
            if is_numeric:
                for col in df.columns:
                    if col[1] == 'values':
                        continue
                    df[col] = df[col].astype(float)

            if baseline_key is not None:
                baseline_scores = df_row.loc[tuple(baseline_key)]

                df2 = df_row.groupby(['model_name', 'feature_name']).agg(list)
                report_columns = []
                for col in df2.columns:
                    report_columns.append(df[[col]])
                    report_columns.append(df2[col].apply(lambda x: get_delta_intervals(col, x, baseline_scores[col])))
                df = pd.concat(report_columns, axis=1)

            if split_columns:
                for mc in main_columns:
                    print(df.loc[:, [mc]], file=self.f)
            else:
                print(df, file=self.f)
            print('', file=self.f)

    def print_footer(self):
        self.print_line()
        _text = f"End of report.     Current time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
        print(_text, file=self.f)
        self.print_line()

    def print_line(self):
        print('-' * 120, file=self.f)
