from functools import reduce
from typing import List

import joblib
import torch
import pandas as pd
from joblib import delayed, Parallel
from collections import ChainMap
from ptls.preprocessing.base.transformation.col_numerical_transformer import ColTransformer


class UserGroupTransformer(ColTransformer):
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

    def __repr__(self):
        return 'Aggregate transformation'

    def _event_time_exist(self, x):
        if self.col_name_original not in x.columns:
            raise AttributeError(f'col_name_original="{self.col_name_original}" not in source dataframe. '
                                 f'Found {x.columns}')
        if 'event_time' not in x.columns:
            raise AttributeError(f'"event_time" not in source dataframe. '
                                 f'Found {x.columns}')

    def _convert_type(self, group_name, group_df):
        group = {}

        for k, v in group_df.to_dict(orient='series').items():
            if k in self.cols_first_item:
                v = v.iloc[0]
            elif isinstance(v.iloc[0], torch.Tensor):
                v = torch.vstack(tuple(v))
            elif v.dtype == 'object':
                v = v.values
            else:
                v = torch.from_numpy(v.values)
            group[k] = v
        group[self.col_name_original] = group_name
        return group

    def fit(self, x):
        self._event_time_exist(x)
        return self

    def df_to_feature_arrays(self, df):
        with joblib.parallel_backend(backend='threading'):
            parallel = Parallel(verbose=1)
            result_dict = parallel(delayed(self._convert_type)(group_name, df_group)
                                   for group_name, df_group in df)

        return result_dict

    def transform(self, x: pd.DataFrame):
        x['et_index'] = x.loc[:, 'event_time']
        x = x.set_index([self.col_name_original, 'et_index']).sort_index().groupby(self.col_name_original)
        x = self.df_to_feature_arrays(x)
        x = x if self.return_records else pd.Series(x)
        return x