import logging
import datetime
import numpy as np
from functools import reduce
from operator import iadd
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from itertools import chain
from typing import List, Dict, Union

from ptls.data_preprocessing.base import DataPreprocessor


logger = logging.getLogger(__name__)


class PysparkDataPreprocessor(DataPreprocessor):
    """Data preprocessor based on pyspark.sql.DataFrame
    During preprocessing it
        * transform `cols_event_time` column with date and time
        * encodes category columns `cols_category` into ints;
        * apply logarithm transformation to `cols_log_norm' columns;
        * (Optional) select the last `max_trx_count` transactions for each `col_id`;
        * groups flat data by `col_id`;
        * arranges data into list of dicts with features
    Parameters
    ----------
    col_id : str
        name of column with ids
    cols_event_time : str,
        name of column with time and date
    cols_category : list[str],s
        list of category columns
    cols_log_norm : list[str],
        list of columns to be logarithmed
    cols_identity : list[str],
        list of columns to be passed as is without any transformation
    cols_target: List[str],
        list of columns with target
    time_transformation: str. Default: 'default'.
        type of transformation to be applied to time column
    remove_long_trx: bool. Default: False.
        If True, select the last `max_trx_count` transactions for each `col_id`.
    max_trx_count: int. Default: 5000.
        used when `remove_long_trx`=True
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
                 remove_long_trx: bool = False,
                 max_trx_count: int = 5000,
                 max_cat_num: Union[Dict[str, int], int] = 10000,
                 print_dataset_info: bool = False):

        super().__init__(col_id, cols_event_time, cols_category, cols_log_norm, cols_identity, cols_target)
        self.time_transformation = time_transformation
        self.time_min = None
        self.remove_long_trx = remove_long_trx
        self.max_trx_count = max_trx_count
        if isinstance(max_cat_num, int):
            self.max_cat_num = {k: max_cat_num for k in cols_category}
        else:
            self.max_cat_num = max_cat_num
        self.print_dataset_info = print_dataset_info


    def fit(self, df, **params):
        """
        Parameters
        ----------
        dt : pyspark.sql.DataFrame with flat data
        Returns
        -------
        self : object
            Fitted preprocessor.
        """
        # Reset internal state before fitting
        self._reset()

        if self.print_dataset_info:
            unique_clients = df.select(self.col_id).distinct().count()
            logger.info(f'Found {unique_clients} unique clients during fit')

        for col in self.cols_category:
            self.cols_category_mapping[col] = self._create_cat_map(df, col)

            if self.print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{self.pd_hist(df, col)}')

        for col in self.cols_log_norm:
            df = self._log_transform(df, col)
            self.cols_log_norm_maxes[col] = df.select(F.max(F.col(col)).alias('max_log1p')).first()[0]

            if self.print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\nlog norm values | trx_count\n{self.pd_hist(df, col)}')

        if self.time_transformation == 'hours_from_min':
            self.time_min = df.select((F.col(self.cols_event_time))\
                                      .cast(dataType=T.TimestampType()).alias('dt'))\
                                      .agg({'dt': 'min'}).collect()[0]['min(dt)']
            self.time_min = (self.time_min - datetime.datetime(1970,1,1)).total_seconds()

        return self


    def transform(self, df, copy=True):
        """Perform preprocessing.
        Parameters
        ----------
        df : pyspark.sql.DataFrame with flat data
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        features : pyspark.sql.DataFrame with all transactions for one `col_id` in one row.
        """
        self.check_is_fitted()
        df_data = df.alias('df_data') if copy else df

        if self.print_dataset_info:
            unique_clients = df.select(self.col_id).distinct().count()
            logger.info(f'Found {unique_clients} unique clients during transform')

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
            df_data = self._map_categories(df_data, col)

            if self.print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{self.pd_hist(df_data, col)}')

        for col in self.cols_log_norm:
            df_data = self._log_transform(df_data, col)
            df_data = df_data.withColumn(col, F.col(col) / self.cols_log_norm_maxes[col])

            if self.print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\nlog norm values | trx_count\n{self.pd_hist(df_data, col)}')

        if self.print_dataset_info:
            df = df_data.groupby(self.col_id).agg(F.count(F.lit(1)).alias("trx_count"))
            logger.info(f'Trx count per clients:\nlen(trx_list) | client_count\n{self.pd_hist(df, "trx_count")}')

        # columns filter
        columns_for_filter = reduce(iadd, [
            self.cols_category,
            self.cols_log_norm,
            self.cols_identity,
            ['event_time', self.col_id],
            self.cols_target,
        ], [])
        used_columns = [col for col in df_data.columns if col in columns_for_filter]

        logger.info('Feature collection in progress ...')
        features = df_data.select(used_columns)
        if self.remove_long_trx:
            features = self._remove_long_trx(features)
        features = self._collect_lists(features)

        features.persist()

        if self.print_dataset_info:
            feature_names = list(features.columns)
            logger.info(f'Feature names: {feature_names}')
            logger.info(f'Prepared features for {features.count()} clients')

        return features


    def _create_cat_map(self, df, col_name):
        df = df.withColumn(col_name, F.coalesce(F.col(col_name).cast('string'), F.lit('#EMPTY')))

        col_orig = '_orig_' + col_name
        df = df.withColumnRenamed(col_name, col_orig)

        df_encoder = df.groupby(col_orig).agg(F.count(F.lit(1)).alias('_cnt'))
        df_encoder = df_encoder.withColumn(col_name,
                                           F.row_number().over(Window.partitionBy().orderBy(F.col('_cnt').desc())))
        df_encoder = df_encoder.filter(F.col(col_name) <= self.max_cat_num[col_name])

        return {row[col_orig]: row[col_name] for row in df_encoder.collect()}


    def _map_categories(self, df, col_name):
        mapping_expr = F.create_map([F.lit(x) for x in chain(*self.cols_category_mapping[col_name].items())])
        df_data = df.withColumn(col_name, mapping_expr[F.col(col_name)])
        val_for_null = max(self.cols_category_mapping[col_name].values())
        df_data = df_data.fillna(value=val_for_null, subset=[col_name])
        return df_data


    @staticmethod
    def _log_transform(df, col_name):
        df = df.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0)))
        df = df.withColumn(col_name, F.signum(F.col(col_name)) * F.log1p(F.abs(F.col(col_name))))
        return df


    def _collect_lists(self, df):
        col_list = [col for col in df.columns if col != self.col_id]

        # if self.config.save_partitioned_data:
        #     df = df.withColumn('mon_id', (F.col('event_time') / 30).cast('int'))
        #     col_id = [col_id, 'mon_id']
        #
        df = df.withColumn('_rn', F.row_number().over(Window.partitionBy(self.col_id).orderBy('event_time')))

        df = df.groupby(self.col_id).agg(*[
            F.sort_array(F.collect_list(F.struct('_rn', col))).alias(col)
            for col in col_list
        ])
        for col in col_list:
            df = df.withColumn(col, F.col(f'{col}.{col}'))

        # df = df.drop('_rn')
        return df


    def _remove_long_trx(self, df):
        """
        This function select the last max_trx_count transactions
        """
        df = df.withColumn('_cn', F.count(F.lit(1)).over(Window.partitionBy(self.col_id)))
        df = df.withColumn('_rn', F.row_number().over(
            Window.partitionBy(self.col_id).orderBy(F.col('event_time').desc())))
        df = df.filter(F.col('_rn') <= self.max_trx_count)
        df = df.drop('_cn')
        df = df.drop('_rn')
        return df


    @staticmethod
    def _td_default(df, cols_event_time):
        w = Window().orderBy(cols_event_time)
        tmp_df = df.select(cols_event_time).distinct()
        tmp_df = tmp_df.withColumn('event_time', F.row_number().over(w) - 1)
        df = df.join(tmp_df, on=cols_event_time)
        return df


    @staticmethod
    def _td_float(df, col_event_time):
        logger.info('To-float time transformation begins...')
        df = df.withColumn('event_time', F.col(col_event_time).astype('float'))
        logger.info('To-float time transformation ends')
        return df


    @staticmethod
    def _td_gender(df, col_event_time):
        """Gender-dataset-like transformation
        'd hh:mm:ss' -> float where integer part is day number and fractional part is seconds from day begin
        '1 00:00:00' -> 1.0
        '1 12:00:00' -> 1.5
        '1 01:00:00' -> 1 + 1 / 24
        '2 23:59:59' -> 1.99
        '432 12:00:00' -> 432.5   '000432 12:00:00'
        :param df:
        :param col_event_time:
        :return:
        """
        logger.info('Gender-dataset-like time transformation begins...')
        df = df.withColumn('_et_day', F.substring(F.lpad(F.col(col_event_time), 15, '0'), 1, 6).cast('float'))

        df = df.withColumn('_et_time', F.substring(F.lpad(F.col(col_event_time), 15, '0'), 8, 8))
        df = df.withColumn('_et_time', F.regexp_replace('_et_time', r'\:60$', ':59'))
        df = df.withColumn('_et_time', F.unix_timestamp('_et_time', 'HH:mm:ss') / (24 * 60 * 60))

        df = df.withColumn('event_time', F.col('_et_day') + F.col('_et_time'))
        df = df.drop('_et_day', '_et_time')
        logger.info('Gender-dataset-like time transformation ends')
        return df


    def _td_hours(self, df, col_event_time):
        logger.info('To hours time transformation begins...')
        df = df.withColumn('_dt', (F.col(col_event_time)).cast(dataType=T.TimestampType()))
        df = df.withColumn('event_time', ((F.col('_dt')).cast('float') - self.time_min) / 3600)
        df = df.drop('_dt')
        logger.info('To hours time transformation ends')
        return df


    def _reset(self):
        """Reset internal data-dependent state of the preprocessor, if necessary.
        __init__ parameters are not touched.
        """
        self.time_min = None
        self.remove_long_trx = False
        self.max_trx_count = 5000
        super()._reset()


    def pd_hist(self, df, name, bins=10):
        # logger.info('pd_hist begin')
        # logger.info(f'sf = {self.config.sample_fraction}')
        data = df.select(name)
        if self.config.sample_fraction is not None:
            data = data.sample(fraction=self.config.sample_fraction)
        data = data.toPandas()[name]

        if data.dtype.kind == 'f':
            round_len = 1 if data.max() > bins + 1 else 2
            bins = np.linspace(data.min(), data.max(), bins + 1).round(round_len)
        elif np.percentile(data, 99) - data.min() > bins - 1:
            bins = np.linspace(data.min(), np.percentile(data, 99), bins).astype(int).tolist() + [int(data.max() + 1)]
        else:
            bins = np.arange(data.min(), data.max() + 2, 1).astype(int)
        df = pd.cut(data, bins, right=False).rename(name)
        df = df.to_frame().assign(cnt=1).groupby(name)[['cnt']].sum()
        df['% of total'] = df['cnt'] / df['cnt'].sum()
        return df

