from sklearn.base import BaseEstimator, TransformerMixin


class ColTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 col_name_original: str,
                 col_name_target: str = None,
                 is_drop_original_col: bool = True,
                 ):
        super().__init__()
        self.col_name_original = col_name_original
        self.col_name_target = col_name_target if col_name_target is not None else col_name_original
        self.is_drop_original_col = is_drop_original_col

    def check_is_col_exists(self, x):
        raise NotImplementedError()

    def drop_original_col(self, x):
        raise NotImplementedError()

    def fit(self, x):
        self.check_is_col_exists(x)
        return self

    def transform(self, x):
        x = self.drop_original_col(x)
        return x
