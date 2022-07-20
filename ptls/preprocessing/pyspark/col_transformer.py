import pyspark.sql

from ptls.preprocessing.base.col_transformer import ColTransformer


class ColTransformerPyspark(ColTransformer):
    def check_is_col_exists(self, x: pyspark.sql.DataFrame):
        if self.col_name_original not in x.columns:
            raise AttributeError(f'cols_event_time="{self.col_name_original}" not in source dataframe. '
                                 f'Found {x.columns}')

    def drop_original_col(self, x: pyspark.sql.DataFrame):
        if self.col_name_original != self.col_name_target and self.is_drop_original_col:
            x = x.drop(self.col_name_original)
        return x
