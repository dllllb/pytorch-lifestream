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
                 is_drop_original_col: bool = False,
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target='event_time',
            is_drop_original_col=is_drop_original_col,
        )

    def __repr__(self):
        return 'Unitary transformation'

    def transform(self, x: pd.Series):
        x = dt_to_timestamp(x)
        new_x = self.attach_column(x)
        return new_x if self.is_drop_original_col else new_x.update({self.col_name_original:x})
