import pandas as pd

from ptls.preprocessing.pandas.col_transformer import ColTransformerPandas


def dt_to_timestamp(x: pd.Series):
    return pd.to_datetime(x).astype('datetime64[s]').astype('int64') // 1000000000


def timestamp_to_dt(x: pd.Series):
    return pd.to_datetime(x, unit='s')


class DatetimeToTimestamp(ColTransformerPandas):
    def __init__(self,
                 col_name_original: str = 'event_time',
                 is_drop_original_col: bool = True,
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target='event_time',
            is_drop_original_col=is_drop_original_col,
        )

    def transform(self, x: pd.DataFrame):
        x[self.col_name_target] = dt_to_timestamp(x[self.col_name_original])
        x = super().transform(x)
        return x
