import pandas as pd

from ptls.preprocessing.base.transformation.col_numerical_transformer import ColTransformer
from ptls.preprocessing.util import dt_to_timestamp


class DatetimeToTimestamp(ColTransformer):
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
        x = self.attach_column(x, dt_to_timestamp(x[self.col_name_original]).rename(self.col_name_target))
        x = super().transform(x)
        return x
