from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class ColIdentityEncoder(ColTransformerPandasMixin, ColTransformer):
    """Dont change original column

    May rename or copy original columns.

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
    def transform(self, x):
        x = x.assign(**{self.col_name_target: lambda x: x[self.col_name_original]})
        x = super().transform(x)
        return x
