from make_datasets_spark import DatasetConverter

import pyspark.sql.functions as F
import logging


FILE_NAME_TRAIN = 'train.csv'
FILE_NAME_TEST = 'test.csv'
COL_EVENT_TIME = 'TRDATETIME'

logger = logging.getLogger(__name__)


class LocalDatasetConverter(DatasetConverter):
    def load_transactions(self):
        df_train = self.spark_read_file(self.path_to_file(FILE_NAME_TRAIN))
        df_test = self.spark_read_file(self.path_to_file(FILE_NAME_TEST))
        logger.info(f'Loaded {df_train.count()} records from "{FILE_NAME_TRAIN}"')
        logger.info(f'Loaded {df_test.count()} records from "{FILE_NAME_TEST}"')

        for col in df_train.columns:
            if col not in df_test.columns:
                df_test = df_test.withColumn(col, F.lit(None))
                logger.info(f'Test extended with "{col}" column')

        df = df_train.union(df_test)

        # event_time mapping
        df = df.withColumn('_et_day', F.substring(F.col(COL_EVENT_TIME), 1, 7))
        df = df.withColumn('_et_day', F.unix_timestamp('_et_day', 'ddMMMyy'))

        df = df.withColumn('_et_time', F.substring(F.col(COL_EVENT_TIME), 9, 8))
        df = df.withColumn('_et_time', F.unix_timestamp('_et_time', 'HH:mm:ss'))

        df = df.withColumn('event_time', F.col('_et_day') + F.col('_et_time'))
        df = df.withColumn('event_time', F.col('event_time') / (24 * 60 * 60))
        df = df.drop('_et_day', '_et_time')

        for col in df.columns:
            df = df.withColumnRenamed(col, col.lower())

        return df

    def load_target(self):
        df_target = self.spark_read_file(self.path_to_file(FILE_NAME_TRAIN))
        f_agg = [F.first(col).alias(col) for col in self.config.col_target]
        df_target = df_target.groupby(self.config.col_client_id).agg(*f_agg)
        return df_target


if __name__ == '__main__':
    LocalDatasetConverter().run()
