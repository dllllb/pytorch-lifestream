import pandas as pd

from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


def dt_to_timestamp(x: pd.Series):
    return pd.to_datetime(x).astype('datetime64[s]').astype('int64') // 1000000000


def timestamp_to_dt(x: pd.Series):
    return pd.to_datetime(x, unit='s')


class DatetimeToTimestamp(ColTransformerPandasMixin, ColTransformer):
    """Converts datetime column to unix timestamp

    Parameters
    ----------
    col_name_original:
        Source column name
    col_name_target:
        Is always `event_time`. This is predefined column name.
    is_drop_original_col:
        When target and original columns are different manage original col deletion.

    """
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
        x = x.assign(**{self.col_name_target: lambda x: dt_to_timestamp(x[self.col_name_original])})
        x = super().transform(x)
        return x
