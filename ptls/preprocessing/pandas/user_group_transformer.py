from typing import List

import torch
import pandas as pd

from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class UserGroupTransformer(ColTransformerPandasMixin, ColTransformer):
    """Groups transactions by user. Splits it by features.

    'event_time' column should be in dataset. We use it to order transactions

    Parameters
    ----------
    col_name_original:
        Column name with user_id - key for grouping
    cols_first_item:
        Only first value will be taken for these columns.
        All values as tensor will be taken for other columns
    return_records:
        False: Result is a dataframe. Use `.to_dict(orient='records')` to transform it to `ptls` format.
        True: Result is a list of dicts - `ptls` format
    """
    def __init__(self,
                 col_name_original: str,
                 cols_first_item: List[str] = None,
                 return_records: bool = False,
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=None,
            is_drop_original_col=False,
        )
        self.cols_first_item = cols_first_item if cols_first_item is not None else []
        self.return_records = return_records

    def fit(self, x):
        # super().fit(x)
        if self.col_name_original not in x.columns:
            raise AttributeError(f'col_name_original="{self.col_name_original}" not in source dataframe. '
                                 f'Found {x.columns}')
        if 'event_time' not in x.columns:
            raise AttributeError(f'"event_time" not in source dataframe. '
                                 f'Found {x.columns}')
        return self

    def df_to_feature_arrays(self, df):
        return pd.Series({k: torch.from_numpy(v.values) if k not in self.cols_first_item else v.iloc[0]
                          for k, v in df.to_dict(orient='series').items()})

    def transform(self, x: pd.DataFrame):
        x = x.assign(et_index=lambda x: x['event_time']).set_index([self.col_name_original, 'et_index'])
        x = x.sort_index().groupby(self.col_name_original)
        x = x.apply(self.df_to_feature_arrays).reset_index()

        # we don't heed to drop original column in this transformer
        # x = super().transform(x)
        if self.return_records:
            x = x.to_dict(orient='records')
        return x
