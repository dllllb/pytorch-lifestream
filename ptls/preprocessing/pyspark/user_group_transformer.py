from typing import List

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Window

from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class UserGroupTransformer(ColTransformerPandasMixin, ColTransformer):
    """Groups transactions by user. Splits it by features.

    'event_time' column should be in dataset. We use it to order transactions

    Parameters
    ----------
    col_name_original:
        Column name with user_id - key for grouping
    cols_last_item:
        Only first value will be taken for these columns.
        All values as tensor will be taken for other columns
    max_trx_count:
        Keep only `max_trx_count` last transactions
    """
    def __init__(self,
                 col_name_original: str,
                 cols_last_item: List[str] = None,
                 max_trx_count: int = 10000,
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=None,
            is_drop_original_col=False,
        )
        self.cols_last_item = cols_last_item if cols_last_item is not None else []
        self.max_trx_count = max_trx_count

    def fit(self, x: pyspark.sql.DataFrame):
        # super().fit(x)
        if self.col_name_original not in x.columns:
            raise AttributeError(f'col_name_original="{self.col_name_original}" not in source dataframe. '
                                 f'Found {x.columns}')
        if 'event_time' not in x.columns:
            raise AttributeError(f'"event_time" not in source dataframe. '
                                 f'Found {x.columns}')
        return self


    def transform(self, x: pyspark.sql.DataFrame):
        col_list = [col for col in x.columns if col != self.col_name_original]

        df = x.withColumn('_rn', F.row_number().over(
            Window.partitionBy(self.col_name_original).orderBy(F.col('event_time').desc())))
        if self.max_trx_count is not None:
            df = df.where(F.col('_rn') <= self.max_trx_count)

        df = df.groupby(self.col_name_original).agg(
            *[
                F.sort_array(F.collect_list(F.struct('_rn', col)), asc=False).alias(col)
                for col in col_list if col not in self.cols_last_item
            ],
            *[
                F.struct(F.max(F.when(F.col('_rn') == 1, F.col(col))).alias(col)).alias(col)
                for col in col_list if col in self.cols_last_item
            ],
        )
        for col in col_list:
            df = df.withColumn(col, F.col(f'{col}.{col}'))

        # we don't heed to drop original column in this transformer
        # x = super().transform(x)
        return df
