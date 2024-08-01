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

    def __init__(self,
                 col_name_original: str = 'event_time',
                 col_name_target: str = 'trans_date',
                 is_drop_original_col: bool = False,
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=col_name_target,
            is_drop_original_col=is_drop_original_col,
        )

    def __repr__(self):
        return 'Unitary transformation'

    def transform(self, x):
        new_x = self.attach_column(x)
        if not self.is_drop_original_col:
            new_x.update({self.col_name_original: x})
        return new_x
