import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import dask.dataframe as dd
from pandas.core.window import Window
from pymonad.either import Either

from ptls.preprocessing.base.data_preprocessor import DataPreprocessor
from ptls.preprocessing.base.transformation.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.base.transformation.col_identity_transformer import ColIdentityEncoder
from ptls.preprocessing.base.transformation.col_numerical_transformer import ColTransformer
from ptls.preprocessing.dask.dask_transformation.category_identity_encoder import CategoryIdentityEncoder
from ptls.preprocessing.dask.dask_transformation.event_time import DatetimeToTimestamp
from ptls.preprocessing.dask.dask_transformation.frequency_encoder import FrequencyEncoder
from ptls.preprocessing.dask.dask_transformation.user_group_transformer import UserGroupTransformer
from ptls.util import OperationParameters

logger = logging.getLogger(__name__)


class DaskDataPreprocessor(DataPreprocessor):
    """Data preprocessor based on dask.dataframe

    During preprocessing it
        * transforms `cols_event_time` column with date and time
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
                 params: Optional[OperationParameters] = None
                 ):
        ###############################################
        # Create dask backend
        self.dask_load_func = {'parquet': dd.read_parquet,
                               'csv': dd.read_csv,
                               'json': dd.read_json}
        self.dask_df = self._create_dask_dataset(params.get('path', None))

        ###############################################
        # Init class attributes
        col_id = params.get('col_id', None)
        cols_category = params.get('cols_category', [])
        cts_category = params.get('cts_category', [])
        cols_numerical = params.get('cols_numerical', [])
        cols_last_item = params.get('cols_identity', [])
        cols_identity = params.get('cols_identity', [])
        ct_event_time = params.get('col_event_time', None)
        evetn_transform = params.get('event_time_transformation', 'dt_to_timestamp')
        category_transformation = params.get('category_transformation', 'frequency')
        max_trx_count = params.get('max_trx_count', None)
        max_cat_num = params.get('max_cat_num', 10000)
        cts_category = params.get('cts_category', [])
        return_records = params.get('return_records',True)
        ###############################################
        # Apply transformation to init columns
        cts_numerical = [ColIdentityEncoder(col_name_original=col) for col in cols_numerical]
        ct_event_time = Either(ct_event_time,
                               monoid=[ct_event_time, evetn_transform == 'dt_to_timestamp']).either(
            left_function=lambda x: ColIdentityEncoder(col_name_original=ct_event_time,
                                                       col_name_target='event_time',
                                                       is_drop_original_col=False),
            right_function=lambda x: DatetimeToTimestamp(col_name_original=params.get('col_event_time', None)))
        t_user_group = UserGroupTransformer(col_name_original=col_id,
                                            cols_last_item=cols_last_item,
                                            max_trx_count=max_trx_count)
        cts_category = []
        cts_category_func = lambda col: Either(col,
                               monoid=[col, category_transformation == 'frequency']).either(
            left_function=lambda x: cts_category.append(x) if type(x) is not str
            else cts_category.append(CategoryIdentityEncoder(col_name_original=x)),
            right_function=lambda x :cts_category.append(FrequencyEncoder(col_name_original=x,
                                                                         max_cat_num=max_cat_num.get(x) if type(max_cat_num) is dict else max_cat_num)))
        cts_category_eval = list(map(cts_category_func,cols_category))




        ###############################################
        # Overload parent class
        super().__init__(
            col_event_time=ct_event_time,
            cols_category=cts_category,
            cols_numerical=cts_numerical,
            cols_identity=cols_identity,
            t_user_group=t_user_group,
            return_records=return_records
        )

    def _create_dask_dataset(self, path):
        for dataset in ['csv', 'json', 'parquet']:
            if path.__contains__(dataset):
                break
            else:
                raise AttributeError
        df = self.dask_load_func[dataset](path)
        return df

    def categorize(self):
        self.dask_df = self.dask_df.categorize(columns=self.dask_df.select_dtypes(include="category").columns.tolist())

    def create_dask_dataset(self):
        self.dask_df = self.dask_df.persist()

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
