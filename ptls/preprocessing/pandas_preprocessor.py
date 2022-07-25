import logging
from typing import List, Union

import numpy as np
import pandas as pd

from .base import DataPreprocessor, ColTransformer
from .base.col_category_transformer import ColCategoryTransformer
from .pandas.category_identity_encoder import CategoryIdentityEncoder
from .pandas.col_identity_transformer import ColIdentityEncoder
from .pandas.event_time import DatetimeToTimestamp
from .pandas.frequency_encoder import FrequencyEncoder
from .pandas.user_group_transformer import UserGroupTransformer

logger = logging.getLogger(__name__)


class PandasDataPreprocessor(DataPreprocessor):
    """Data preprocessor based on pandas.DataFrame

    During preprocessing it
        * transform datetime column to `event_time`
        * encodes category columns into indexes;
        * groups flat data by `col_id`;
        * arranges data into list of dicts with features

    Preprocessor don't modify original dataframe, but links to his data.

    Parameters
    ----------
    col_id : str
        name of column with ids. Used for groups
    col_event_time : str
        name of column with datetime
        or `ColTransformer` implementation with datetime transformation
    event_time_transformation: str
        name of transformation for `col_event_time`
        - 'dt_to_timestamp': datetime (string of datetime64) to timestamp (long) with `DatetimeToTimestamp`
            Original column is dropped by default cause target col `event_time` is the same information
            and we can not use as feature datetime column itself.
        - 'none': without transformation, `col_event_time` is in correct format. Used `ColIdentityEncoder`
            Original column is kept by default cause it can be any type and we may use it in the future
    cols_category : list[str]
        list of category columns. Each can me column name or `ColCategoryTransformer` implementation.
    category_transformation: str
        name of transformation for column names from `cols_category`
        - 'frequency': frequency encoding with `FrequencyEncoder`
        - 'none': no transformation with `CategoryIdentityEncoder`
    cols_numerical : list[str]
        list of columns to be mentioned as numerical features. No transformation with `ColIdentityEncoder`
    cols_identity : list[str]
        list of columns to be passed as is without any transformation
    cols_first_item: List[str]
        Only first value will be taken for these columns
        It can be user-level information joined to each transaction
    return_records:
        False: Result is a `pandas.DataFrame`.
            You can:
            - join any additional information like user-level features of target
            - convert it to `ptls` format using `.to_dict(orient='records')`
        True: Result is a list of dicts - `ptls` format

    """

    def __init__(self,
                 col_id: str,
                 col_event_time: Union[str, ColTransformer],
                 event_time_transformation: str = 'dt_to_timestamp',
                 cols_category: List[Union[str, ColCategoryTransformer]] = None,
                 category_transformation: str = 'frequency',
                 cols_numerical: List[str] = None,
                 cols_identity: List[str] = None,
                 cols_first_item: List[str] = None,
                 return_records: bool = True,
                 ):
        if cols_category is None:
            cols_category = []
        if cols_numerical is None:
            cols_numerical = []
        if cols_identity is None:
            cols_identity = []
        if cols_first_item is None:
            cols_first_item = []

        if type(col_event_time) is not str:
            ct_event_time = col_event_time  # use as is
        elif event_time_transformation == 'dt_to_timestamp':
            ct_event_time = DatetimeToTimestamp(col_name_original=col_event_time)
        elif event_time_transformation == 'none':
            ct_event_time = ColIdentityEncoder(
                col_name_original=col_event_time,
                col_name_target='event_time',
                is_drop_original_col=False,
            )
        else:
            raise AttributeError(f'incorrect event_time parameters combination: '
                                 f'`ct_event_time` = "{col_event_time}" '
                                 f'`event_time_transformation` = "{event_time_transformation}"')

        cts_category = []
        for col in cols_category:
            if type(col) is not str:
                cts_category.append(col)  # use as is
            elif category_transformation == 'frequency':
                cts_category.append(FrequencyEncoder(col_name_original=col))
            elif category_transformation == 'none':
                cts_category.append(CategoryIdentityEncoder(col_name_original=col))
            else:
                raise AttributeError(f'incorrect category parameters combination: '
                                     f'`cols_category[i]` = "{col}" '
                                     f'`category_transformation` = "{category_transformation}"')

        cts_numerical = [ColIdentityEncoder(col_name_original=col) for col in cols_numerical]
        t_user_group = UserGroupTransformer(
            col_name_original=col_id, cols_first_item=cols_first_item, return_records=return_records)

        super().__init__(
            ct_event_time=ct_event_time,
            cts_category=cts_category,
            cts_numerical=cts_numerical,
            cols_identity=cols_identity,
            t_user_group=t_user_group,
        )

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
