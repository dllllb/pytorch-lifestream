from functools import reduce
from operator import iadd
from typing import List

from pymonad.maybe import Maybe
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
        self._fit_transform_operation = lambda operation: operation.fit_transform
        self._transform_operation = lambda operation: operation.transform

    # def _preproc_pipeline(self, X, y=None, transform_operation = None, **fit_params):
    #     transformed_features = Maybe(value=X, monoid=True) \
    #         .then(function=lambda x: list(map(transform_operation(x), self.cols_identity))) \
    #         .then(function=lambda x: list(map(transform_operation, self.ct_event_time))) \
    #         .then(function=lambda x: list(map(transform_operation, self.cts_numerical))) \
    #         .then(function=lambda x: list(map(transform_operation, self.cts_category))).value
    #
    #     return transformed_features

    def fit(self, x):
        for i, ct in enumerate(self._all_col_transformers):
            if i == len(self._all_col_transformers):
                ct.fit(x)
            else:
                x = ct.fit_transform(x)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        transformed_features = Maybe(value=X, monoid=True) \
            .then(function=lambda x: list(map(lambda operation: operation.fit_transform(x), self.cols_identity))) \
            .then(function=lambda x: list(map(lambda operation: operation.fit_transform(x), self.ct_event_time))) \
            .then(function=lambda x: list(map(lambda operation: operation.fit_transform(x), self.cts_numerical))) \
            .then(function=lambda x: list(map(lambda operation: operation.fit_transform(x), self.cts_category))).value

        return transformed_features

    def transform(self, X):
        transformed_features = Maybe(value=X, monoid=True) \
            .then(function=lambda x: list(map(lambda operation: operation.fit_transform(x), self.cols_identity))) \
            .then(function=lambda x: list(map(lambda operation: operation.fit_transform(x), self.ct_event_time))) \
            .then(function=lambda x: list(map(lambda operation: operation.fit_transform(x), self.cts_numerical))) \
            .then(function=lambda x: list(map(lambda operation: operation.fit_transform(x), self.cts_category))).value

        return transformed_features

    def get_category_dictionary_sizes(self):
        """Gets a dict of mapping to integers lengths for categories
        """
        return {ct.col_name_target: ct.dictionary_size for ct in self.cts_category}

    def to_yaml(self):
        raise NotImplementedError()

    def from_yaml(self):
        raise NotImplementedError()
