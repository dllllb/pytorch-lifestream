import logging
from typing import List, Union

import numpy as np
import pandas as pd

from ptls.preprocessing.base.data_preprocessor import DataPreprocessor
from ptls.preprocessing.base.transformation.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.base.transformation.col_numerical_transformer import ColTransformer

logger = logging.getLogger(__name__)


class PandasDataPreprocessor(DataPreprocessor):
    """Data preprocessor based on pandas.DataFrame

    During preprocess it
        * transforms datetime column to `event_time`
        * encodes category columns into indexes;
        * groups flat data by `col_id`;
        * arranges data into list of dicts with features

    Preprocessor don't modify original dataframe, but links to his data.

    Args:
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
            list of category columns. Each can be column name or `ColCategoryTransformer` implementation.
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
        n_jobs:
            Number of workers requested by the callers.
            Passing n_jobs=-1 means requesting all available workers for instance matching the number of
            CPU cores on the worker host(s).
    """

    def __init__(
        self,
        col_id: str,
        col_event_time: Union[str, ColTransformer],
        event_time_transformation: str = "dt_to_timestamp",
        cols_category: List[Union[str, ColCategoryTransformer]] = None,
        category_transformation: str = "frequency",
        cols_numerical: List[str] = None,
        cols_identity: List[str] = None,
        cols_first_item: List[str] = None,
        return_records: bool = True,
        n_jobs: int = -1,
    ):
        self.category_transformation = category_transformation
        self.return_records = return_records
        self.cols_first_item = cols_first_item
        self.event_time_transformation = event_time_transformation
        self.n_jobs = n_jobs
        super().__init__(
            col_id=col_id,
            col_event_time=col_event_time,
            cols_category=cols_category,
            cols_identity=cols_identity,
            cols_numerical=cols_numerical,
            n_jobs=n_jobs,
        )

    @staticmethod
    def _td_default(df, cols_event_time):
        df_event_time = df[cols_event_time].drop_duplicates()
        df_event_time = df_event_time.sort_values(cols_event_time)
        df_event_time["event_time"] = np.arange(len(df_event_time))
        df = pd.merge(df, df_event_time, on=cols_event_time)
        logger.info("Default time transformation")
        return df

    @staticmethod
    def _td_float(df, col_event_time):
        df["event_time"] = df[col_event_time].astype(float)
        logger.info("To-float time transformation")
        return df

    @staticmethod
    def _td_gender(df: pd.DataFrame, col_event_time: str):
        """Gender-dataset-like transformation

        'd hh:mm:ss' -> float where integer part is day number and fractional part is seconds from day begin
        '1 00:00:00' -> 1.0
        '1 12:00:00' -> 1.5
        '1 01:00:00' -> 1 + 1 / 24
        '2 23:59:59' -> 1.99
        '432 12:00:00' -> 432.5

        Args:
            df: DataFrame
            col_event_time: name of column with datetime

        """
        padded_time = df[col_event_time].str.pad(15, "left", "0")
        day_part = padded_time.str[:6].astype(float)
        time_part = (
            pd.to_datetime(padded_time.str[7:], format="%H:%M:%S").values.astype(int)
            // 1e9
        )
        time_part = time_part % (24 * 60 * 60) / (24 * 60 * 60)
        df["event_time"] = day_part + time_part
        logger.info("Gender-dataset-like time transformation")
        return df

    def _td_hours(self, df, col_event_time):
        logger.info("To hours time transformation")
        df["event_time"] = pd.to_datetime(df[col_event_time])
        df["event_time"] = (df["event_time"] - self.time_min).dt.total_seconds() / 3600
        return df
