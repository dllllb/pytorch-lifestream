import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColTransformer(BaseEstimator, TransformerMixin):
    """Base class for dataframe column transformer. Check if columns exists.
    May rename or copy original column

    Args:
        col_name_original: Source column name
        col_name_target: Target column name. Transformed column will be placed here
            If `col_name_target is None` then original column will be replaced by transformed values.
        is_drop_original_col: When target and original columns are different manage original col deletion.

    """

    def __init__(
        self,
        col_name_original: str,
        col_name_target: str = None,
        is_drop_original_col: bool = True,
    ):
        super().__init__()
        self.col_name_original = col_name_original
        self.col_name_target = (
            col_name_target if col_name_target is not None else col_name_original
        )
        self.is_drop_original_col = is_drop_original_col

    def __repr__(self):
        return "Unitary transformation"

    def check_is_col_exists(self, x: pd.DataFrame):
        if x is None:
            raise AttributeError(
                f'col_name_original="{self.col_name_original}" not in source dataframe.'
            )

    def drop_original_col(self, x: pd.DataFrame):
        if self.col_name_original != self.col_name_target and self.is_drop_original_col:
            x = x.drop(columns=self.col_name_original)
        return x

    # def _attach_column(self, df: pd.Series):
    #     return {self.col_name_target: df}

    @staticmethod
    def attach_column(df: pd.DataFrame, s: pd.Series):
        new_col_name = s.name
        cols = [col for col in df.columns if col != new_col_name]
        return pd.concat([df[cols], s], axis=1)

    def fit(self, x):
        self.check_is_col_exists(x)
        return self

    def transform(self, x):
        # new_col = self.attach_column(x)
        # del x
        # return new_col
        return self.drop_original_col(x)
