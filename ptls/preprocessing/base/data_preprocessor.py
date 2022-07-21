from functools import reduce
from operator import iadd
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin

from .col_category_transformer import ColCategoryTransformer
from .col_transformer import ColTransformer


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 ct_event_time: ColTransformer,
                 cts_category: List[ColCategoryTransformer],
                 cts_numerical: List[ColTransformer],
                 cols_identity: List[str],
                 t_user_group: ColTransformer,
                 ):
        self.ct_event_time = ct_event_time
        self.cts_category = cts_category
        self.cts_numerical = cts_numerical
        self.cols_identity = cols_identity
        self.t_user_group = t_user_group

        self._all_col_transformers = [
            [self.ct_event_time],
            self.cts_category,
            self.cts_numerical,
            [self.t_user_group],
        ]
        self._all_col_transformers = reduce(iadd, self._all_col_transformers, [])

    def fit(self, x):
        for i, ct in enumerate(self._all_col_transformers):
            if i == len(self._all_col_transformers):
                ct.fit(x)
            else:
                x = ct.fit_transform(x)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        for ct in self._all_col_transformers:
            X = ct.fit_transform(X)
        return X

    def transform(self, x):
        for ct in self._all_col_transformers:
            x = ct.transform(x)
        return x

    def get_category_dictionary_sizes(self):
        """Gets a dict of mapping to integers lengths for categories
        """
        return {ct.col_name_target: ct.dictionary_size for ct in self.cts_category}

    def to_yaml(self):
        raise NotImplementedError()

    def from_yaml(self):
        raise NotImplementedError()
