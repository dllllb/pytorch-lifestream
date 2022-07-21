import pyspark.sql
from pyspark.sql import functions as F

from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pyspark.col_transformer import ColTransformerPysparkMixin


def dt_to_timestamp(col: str):
    return F.unix_timestamp(F.col(col).cast('timestamp'))


def timestamp_to_dt(col: str):
    return F.from_unixtime(col).cast('timestamp')


class DatetimeToTimestamp(ColTransformerPysparkMixin, ColTransformer):
    def __init__(self,
                 col_name_original: str = 'event_time',
                 is_drop_original_col: bool = True,
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target='event_time',
            is_drop_original_col=is_drop_original_col,
        )

    def transform(self, x: pyspark.sql.DataFrame):
        x = x.withColumn(self.col_name_target, dt_to_timestamp(self.col_name_original))
        x = super().transform(x)
        return x
