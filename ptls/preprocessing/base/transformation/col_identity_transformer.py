from ptls.preprocessing.base.transformation.col_numerical_transformer import ColTransformer


class ColIdentityEncoder(ColTransformer):
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
    def __repr__(self):
        return 'Unitary transformation'

    def transform(self, x):
        return super().transform(x)
