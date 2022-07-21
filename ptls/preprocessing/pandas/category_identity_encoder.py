import warnings

import pandas as pd

from ptls.preprocessing.base.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class CategoryIdentityEncoder(ColTransformerPandasMixin, ColCategoryTransformer):
    """Keep encoding from original category column

    Let's `col_name_original` value_counts looks like this:
    cat value: records counts in dataset
          4:  100
          2:  50
          1:  10
          6:  1

    Mapping will use this order to enumerate embedding indexes for category values:
    cat value: embedding id
    <padding token>: 0
                1: 1
                2: 2
                4: 4
                6: 6
     <other values>: 6

    `dictionary_size` will be 7

    Note:
       - expect integer values in original column which are mentioned as embedding indexes
       - 0 index is reserved for padding value
       - negative indexes aren't allowed
       - there are no <other values>. Input and output are identical.

    Parameters
    ----------
    col_name_original:
        Source column name
    col_name_target:
        Target column name. Transformed column will be placed here
        If `col_name_target is None` then original column will be replaced by transformed values.
    is_drop_original_col:
        When target and original columns are different manage original col deletion.

    """
    def __init__(self,
                 col_name_original: str,
                 col_name_target: str = None,
                 is_drop_original_col: bool = True,
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=col_name_target,
            is_drop_original_col=is_drop_original_col,
        )

        self.min_fit_index = None
        self.max_fit_index = None

    def get_column(self, x):
        return x[self.col_name_original].astype(int)

    def fit(self, x: pd.DataFrame):
        super().fit(x)
        pd_col = self.get_column(x)
        self.min_fit_index, self.max_fit_index = pd_col.agg([min, max])
        if self.min_fit_index < 0:
            raise AttributeError(f'Negative values found in {self.col_name_original}')
        if self.min_fit_index == 0:
            warnings.warn(f'0 values fount in {self.col_name_original}. 0 is a padding index', UserWarning)
        return self

    @property
    def dictionary_size(self):
        return self.max_fit_index + 1

    def transform(self, x: pd.DataFrame):
        pd_col = self.get_column(x)
        x = x.assign(**{self.col_name_target: pd_col})

        min_index, max_index = pd_col.agg([min, max])
        if min_index < self.min_fit_index:
            warnings.warn(f'Not fitted values. min_index({min_index}) < min_fit_index({self.min_fit_index})',
                          UserWarning)
        if max_index > self.max_fit_index:
            warnings.warn(f'Not fitted values. max_index({max_index}) < max_fit_index({self.max_fit_index})',
                          UserWarning)

        x = super().transform(x)
        return x
