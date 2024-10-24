from typing import List

import pyspark
import pyspark.sql.functions as F

from ptls.preprocessing.base.transformation.col_numerical_transformer import ColTransformer
from ptls.preprocessing.dask.dask_transformation.col_transformer import ColTransformerPandasMixin


class UserGroupTransformer(ColTransformerPandasMixin, ColTransformer):
    """Groups transactions by user. Splits it by features.

    'event_time' column should be in dataset. We use it to order transactions

    Args:
        col_name_original: Column name with user_id - key for grouping
        cols_last_item: Only first value will be taken for these columns.
            All values as tensor will be taken for other columns
        max_trx_count: Keep only `max_trx_count` last transactions

    """

    def __init__(
        self,
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
            raise AttributeError(
                f'col_name_original="{self.col_name_original}" not in source dataframe. '
                f"Found {x.columns}"
            )
        if "event_time" not in x.columns:
            raise AttributeError(
                f'"event_time" not in source dataframe. ' f"Found {x.columns}"
            )
        return self

    def transform(self, x: pyspark.sql.DataFrame):
        col_list = ["event_time"] + [
            col
            for col in x.columns
            if col != self.col_name_original and col != "event_time"
        ]
        unpack_col_list = [self.col_name_original] + [
            F.col(f"_struct.{col}").alias(col) for col in col_list
        ]

        # Put columns into structs and collect structs.
        df = x.groupBy(self.col_name_original).agg(
            F.sort_array(F.collect_list(F.struct(*col_list))).alias("_struct")
        )
        if self.max_trx_count is not None:
            array_slice = F.slice(
                F.col("_struct"),
                F.greatest(
                    F.size(F.col("_struct")) - (self.max_trx_count - 1), F.lit(1)
                ),
                self.max_trx_count,
            )
            df = df.withColumn("_struct", array_slice)

        # Unpack structs.
        df = df.select(*unpack_col_list)

        # Select last elements.
        for col in self.cols_last_item:
            df = df.withColumn(col, F.element_at(F.col(col), -1))

        return df
