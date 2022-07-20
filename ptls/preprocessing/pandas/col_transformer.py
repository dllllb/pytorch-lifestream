import pandas as pd

from ptls.preprocessing.base.col_transformer import ColTransformer


class ColTransformerPandas(ColTransformer):
    def check_is_col_exists(self, x: pd.DataFrame):
        if self.col_name_original not in x.columns:
            raise AttributeError(f'cols_event_time="{self.col_name_original}" not in source dataframe. '
                                 f'Found {x.columns}')

    def drop_original_col(self, x: pd.DataFrame):
        if self.col_name_original != self.col_name_target and self.is_drop_original_col:
            x = x[[col for col in x.columns if col != self.col_name_original]]
        return x
