from itertools import chain

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Window

from ptls.preprocessing.base.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.pyspark.col_transformer import ColTransformerPysparkMixin


class FrequencyEncoder(ColTransformerPysparkMixin, ColCategoryTransformer):
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
    max_cat_num:
        Maximum category number
    """
    def __init__(self,
                 col_name_original: str,
                 col_name_target: str = None,
                 is_drop_original_col: bool = True,
                 max_cat_num: int = 10000,
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=col_name_target,
            is_drop_original_col=is_drop_original_col,
        )

        self.mapping = None
        self.other_values_code = None
        self.max_cat_num = max_cat_num

    def get_col(self, x: pyspark.sql.DataFrame):
        return x.withColumn(self.col_name_target,
                          F.coalesce(F.col(self.col_name_original).cast('string'), F.lit('#EMPTY')))

    def fit(self, x: pyspark.sql.DataFrame):
        super().fit(x)

        df = self.get_col(x)
        df_encoder = df.groupby(self.col_name_target).agg(F.count(F.lit(1)).alias('_cnt'))
        df_encoder = df_encoder.withColumn('_rn',
                                           F.row_number().over(Window.partitionBy().orderBy(F.col('_cnt').desc())))
        df_encoder = df_encoder.filter(F.col('_rn') <= self.max_cat_num)

        self.mapping = {row[self.col_name_target]: row['_rn'] for row in df_encoder.collect()}
        self.other_values_code = len(self.mapping) + 1
        return self

    @property
    def dictionary_size(self):
        return self.other_values_code + 1

    def transform(self, x: pyspark.sql.DataFrame):
        df = self.get_col(x)

        mapping_expr = F.create_map([F.lit(x) for x in chain(*self.mapping.items())])
        df = df.withColumn(self.col_name_target, mapping_expr[F.col(self.col_name_target)])
        df = df.fillna(value=self.other_values_code, subset=[self.col_name_target])

        x = super().transform(df)
        return x
