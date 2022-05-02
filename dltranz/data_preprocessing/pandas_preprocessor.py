import logging
from functools import reduce
from operator import iadd
from typing import List

import numpy as np
import pandas as pd

from typing import List

from .base import DataPreprocessor
from .util import pd_hist

logger = logging.getLogger(__name__)


class PandasDataPreprocessor(DataPreprocessor):
    """Data preprocessor based on pandas.DataFrame

    During preprocessing it
        * transform `cols_event_time` column with date and time
        * encodes category columns `cols_category` into ints;
        * apply logarithm transformation to `cols_log_norm' columns;
        * groups flat data by `col_id`;
        * arranges data into list of dicts with features

    Parameters
    ----------
    col_id : str
        name of column with ids
    cols_event_time : str,
        name of column with time and date
    cols_category : list[str],
        list of category columns
    cols_log_norm : list[str],
        list of columns to be logarithmed
    cols_identity : list[str],
        list of columns to be passed as is without any transformation
    cols_target: List[str],
        list of columns with target
    time_transformation: str. Default: 'default'.
        type of transformation to be applied to time column
    print_dataset_info : bool. Default: False.
        If True, print dataset stats during preprocessor fitting and data transformation
    """

    def __init__(self,
                 col_id: str,
                 cols_event_time: str,
                 cols_category: List[str],
                 cols_log_norm: List[str],
                 cols_identity: List[str] = [],
                 cols_target: List[str] = [],
                 time_transformation: str = 'default',
                 print_dataset_info: bool = False,
                 ):

        super().__init__(col_id, cols_event_time, cols_category, cols_log_norm, cols_identity, cols_target)
        
        self.print_dataset_info = print_dataset_info
        self.time_transformation = time_transformation
        self.time_min = None

    def fit(self, dt, **params):
        """
        Parameters
        ----------
        dt : pandas.DataFrame with flat data

        Returns
        -------
        self : object
            Fitted preprocessor.
        """
        # Reset internal state before fitting
        self._reset()

        for col in self.cols_category:
            pd_col = dt[col].astype(str)
            mapping = {k: i + 1 for i, k in enumerate(pd_col.value_counts().index)}
            self.cols_category_mapping[col] = mapping

            if self.print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{pd_hist(dt[col], col)}')

        for col in self.cols_log_norm:
            self.cols_log_norm_maxes[col] = (np.log1p(abs(dt[col])) * np.sign(dt[col])).max()

        if self.time_transformation == 'hours_from_min':
            self.time_min = pd.to_datetime(dt[self.cols_event_time]).min()

        return self

    def transform(self, df, copy=True):
        """Perform preprocessing.
        Parameters
        ----------
        df : pandas.DataFrame with flat data
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        features : List of dicts grouped by col_id.
        """
        self.check_is_fitted()
        df_data = df.copy() if copy else df

        if self.print_dataset_info:
            logger.info(f'Found {df_data[self.col_id].nunique()} unique ids')

        # event_time mapping
        if self.time_transformation == 'none':
            pass
        elif self.time_transformation == 'default':
            df_data = self._td_default(df_data, self.cols_event_time)
        elif self.time_transformation == 'float':
            df_data = self._td_float(df_data, self.cols_event_time)
        elif self.time_transformation == 'gender':
            df_data = self._td_gender(df_data, self.cols_event_time)
        elif self.time_transformation == 'hours_from_min':
            df_data = self._td_hours(df_data, self.cols_event_time)
        else:
            raise NotImplementedError(f'Unknown type of data transformation: "{self.time_transformation}"')

        for col in self.cols_category:
            if col not in self.cols_category_mapping:
                raise KeyError(f"column {col} isn't in fitted category columns")
            pd_col = df_data[col].astype(str)
            df_data[col] = pd_col.map(self.cols_category_mapping[col]) \
                .fillna(max(self.cols_category_mapping[col].values()))
            if self.print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{pd_hist(df_data[col], col)}')

        for col in self.cols_log_norm:
            df_data[col] = np.log1p(abs(df_data[col])) * np.sign(df_data[col])
            df_data[col] /= self.cols_log_norm_maxes[col]
            if self.print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{pd_hist(df_data[col], col)}')

        if self.print_dataset_info:
            df = df_data.groupby(self.col_id)['event_time'].count()
            logger.info(f'Trx count per clients:\nlen(trx_list) | client_count\n{pd_hist(df, "trx_count")}')

        # column filter
        columns_for_filter = reduce(iadd, [
            self.cols_category,
            self.cols_log_norm,
            self.cols_identity,
            ['event_time', self.col_id],
            self.cols_target,
        ], [])
        used_columns = [col for col in df_data.columns if col in columns_for_filter]

        logger.info('Feature collection in progress ...')
        features = df_data[used_columns] \
            .assign(et_index=lambda x: x['event_time']) \
            .set_index([self.col_id, 'et_index']).sort_index() \
            .groupby(self.col_id).apply(lambda x: {k: v[0] if k in self.cols_target else np.array(v)
                                                   for k, v in x.to_dict(orient='list').items()}) \
            .rename('feature_arrays').reset_index().to_dict(orient='records')

        def squeeze(rec):
            return {self.col_id: rec[self.col_id], **rec['feature_arrays']}
        features = [squeeze(r) for r in features]

        if self.print_dataset_info:
            feature_names = list(features[0].keys())
            logger.info(f'Feature names: {feature_names}')

        logger.info(f'Prepared features for {len(features)} clients')
        return features

    def _reset(self):
        """Reset internal data-dependent state of the preprocessor, if necessary.
        __init__ parameters are not touched.
        """
        self.time_min = None
        super()._reset()

    @staticmethod
    def _td_default(df, cols_event_time):
        df_event_time = df[cols_event_time].drop_duplicates()
        df_event_time = df_event_time.sort_values(cols_event_time)
        df_event_time['event_time'] = np.arange(len(df_event_time))
        df = pd.merge(df, df_event_time, on=cols_event_time)
        logger.info('Default time transformation')
        return df

    @staticmethod
    def _td_float(df, col_event_time):
        df['event_time'] = df[col_event_time].astype(float)
        logger.info('To-float time transformation')
        return df

    @staticmethod
    def _td_gender(df, col_event_time):
        """Gender-dataset-like transformation

        'd hh:mm:ss' -> float where integer part is day number and fractional part is seconds from day begin
        '1 00:00:00' -> 1.0
        '1 12:00:00' -> 1.5
        '1 01:00:00' -> 1 + 1 / 24
        '2 23:59:59' -> 1.99
        '432 12:00:00' -> 432.5

        :param df:
        :param col_event_time:
        :return:
        """
        padded_time = df[col_event_time].str.pad(15, 'left', '0')
        day_part = padded_time.str[:6].astype(float)
        time_part = pd.to_datetime(padded_time.str[7:], format='%H:%M:%S').values.astype(int) // 1e9
        time_part = time_part % (24 * 60 * 60) / (24 * 60 * 60)
        df['event_time'] = day_part + time_part
        logger.info('Gender-dataset-like time transformation')
        return df

    def _td_hours(self, df, col_event_time):
        logger.info('To hours time transformation')
        df['event_time'] = pd.to_datetime(df[col_event_time])
        df['event_time'] = (df['event_time'] - self.time_min).dt.total_seconds() / 3600

        return df
