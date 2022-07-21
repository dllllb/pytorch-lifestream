import pandas as pd

from ptls.preprocessing.base.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class FrequencyEncoder(ColTransformerPandasMixin, ColCategoryTransformer):
    """Use frequency encoding for categorical field

    Let's `col_name_original` value_counts looks like this:
    cat value: records counts in dataset
          aaa:  100
          bbb:  50
          nan:  10
          ccc:  1

    Mapping will use this order to enumerate embedding indexes for category values:
    cat value: embedding id
    <padding token>: 0
                aaa: 1
                bbb: 2
                nan: 3
                ccc: 4
     <other values>: 5

    `dictionary_size` will be 6

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

        self.mapping = None
        self.other_values_code = None

    def fit(self, x: pd.DataFrame):
        super().fit(x)
        pd_col = x[self.col_name_original].astype(str)
        vc = pd_col.value_counts()
        self.mapping = {k: i + 1 for i, k in enumerate(vc.index)}
        self.other_values_code = len(vc) + 1
        return self

    @property
    def dictionary_size(self):
        return self.other_values_code + 1

    def transform(self, x: pd.DataFrame):
        pd_col = x[self.col_name_original].astype(str)
        x = x.assign(**{self.col_name_target: pd_col.map(self.mapping).fillna(self.other_values_code)})
        x = super().transform(x)
        return x
