import pyspark.sql.functions as F

from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pyspark.col_transformer import ColTransformerPysparkMixin


class ColIdentityEncoder(ColTransformerPysparkMixin, ColTransformer):
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
        x = x.withColumn(self.col_name_target, F.col(self.col_name_original))
        x = super().transform(x)
        return x
