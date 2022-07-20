from typing import List

from sklearn.base import BaseEstimator, TransformerMixin

from .col_transformer import ColTransformer


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 col_id: str,
                 col_event_time: ColTransformer,
                 cols_category: List[ColTransformer],
                 cols_numerical: List[ColTransformer],
                 cols_identity: List[ColTransformer],
                 cols_target: List[str],
                 ):
        self.col_id = col_id
        self.col_event_time = col_event_time
        self.cols_category = cols_category
        self.cols_numerical = cols_numerical
        self.cols_identity = cols_identity
        self.cols_target = cols_target

    def get_category_sizes(self):
        """Gets a dict of mapping to integers lengths for categories
        """
        return {k: len(v) for k, v in self.cols_category_mapping.items()}

    def to_yaml(self):
        raise NotImplementedError()

    def from_yaml(self):
        raise NotImplementedError()
