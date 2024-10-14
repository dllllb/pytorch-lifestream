from ptls.make_datasets_spark import DatasetConverter

import pyspark.sql.functions as F
import logging
from glob import glob
import os
import datetime

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql import Window

logger = logging.getLogger(__name__)


class LocalDatasetConverter(DatasetConverter):
    FILE_NAME_TRAIN = 'train_transactions_contest.parquet'
    FILE_NAME_TEST = 'test_transactions_contest.parquet'
    COL_EVENT_TIME = 'transaction_number'
    FILE_NAME_TARGET = 'train_target.csv'

    def load_transactions(self):
        df_train = self.spark_read_file(self.path_to_file(self.FILE_NAME_TRAIN))
        df_test = self.spark_read_file(self.path_to_file(self.FILE_NAME_TEST))
        logger.info(f'Loaded {df_train.count()} records from "{self.FILE_NAME_TRAIN}"')
        logger.info(f'Loaded {df_test.count()} records from "{self.FILE_NAME_TEST}"')

        for col in df_train.columns:
            if col not in df_test.columns:
                df_test = df_test.withColumn(col, F.lit(None))
                logger.info(f'Test extended with "{col}" column')

        df = df_train.union(df_test)

        # event_time mapping
        df = df.withColumn('event_time', F.col(self.COL_EVENT_TIME))

        for col in df.columns:
            df = df.withColumnRenamed(col, col.lower())

        return df

    def load_target(self):
        df_target = self.spark_read_file(self.path_to_file(self.FILE_NAME_TARGET))
        df_target = df_target.withColumn('flag', F.col('flag').cast('int'))
        return df_target

    def run(self):
        _start = datetime.datetime.now()
        self.parse_args()
        self.logging_config()

        df = self.load_target()
        ws = Window.partitionBy('product', 'flag')
        df = df.withColumn('_hash', F.hash(
            F.concat(F.col(self.config.col_client_id), F.lit(self.config.salt))) / 2**32 + 0.5)
        df = df.withColumn('p', F.row_number().over(ws.orderBy('_hash')) / F.count('*').over(ws))

        df_target_train = df.where(F.col('p') >= self.config.test_size).drop('_hash', 'p')
        df_target_test = df.where(F.col('p') < self.config.test_size).drop('_hash', 'p')
        df_target_train.persist()
        df_target_test.persist()

        logger.info(f'{df_target_train.count()} apps in train, {df_target_test.count()} in test')

        self.save_test_ids(df_target_test)

        logger.info(f'Start processing')
        for path in sorted(glob(self.path_to_file(self.FILE_NAME_TRAIN) + '/*.parquet')):
            self.process_train_partition(df_target_test, df_target_train, 'train', path)
        for path in sorted(glob(self.path_to_file(self.FILE_NAME_TEST) + '/*.parquet')):
            self.process_test_partition(df_target_train, 'test', path)

        _duration = datetime.datetime.now() - _start
        logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')

    def process_train_partition(self, df_target_test, df_target_train, p_name, path):
        spark = SparkSession.builder.getOrCreate()
        df_trx, file_name = self.get_trx(path)

        df_trx_train = df_trx.join(
            df_target_test.select(self.config.col_client_id),
            on=self.config.col_client_id,
            how='left_anti',
        ).join(
            df_target_train,
            on=self.config.col_client_id,
            how='left',
        )
        df_trx_test = df_trx.join(
            df_target_test,
            on=self.config.col_client_id,
            how='inner',
        )
        train_path_to = self.config.output_train_path + f'/{p_name}_{file_name}'
        test_path_to = self.config.output_test_path + f'/{p_name}_{file_name}'
        df_trx_train.write.parquet(train_path_to, mode='overwrite')
        df_trx_test.write.parquet(test_path_to, mode='overwrite')
        df_trx_train = spark.read.parquet(train_path_to)
        df_trx_test = spark.read.parquet(test_path_to)
        logger.info(f'{p_name}: {file_name} - done. '
                    f'Train: {df_trx_train.count()}, test: {df_trx_test.count()}')

    def process_test_partition(self, df_target_train, p_name, path):
        spark = SparkSession.builder.getOrCreate()
        df_trx, file_name = self.get_trx(path)

        df_trx_train = df_trx.join(
            df_target_train,
            on=self.config.col_client_id,
            how='left',
        )
        train_path_to = self.config.output_train_path + f'/{p_name}_{file_name}'
        df_trx_train.write.parquet(train_path_to, mode='overwrite')
        df_trx_train = spark.read.parquet(train_path_to)
        logger.info(f'{p_name}: {file_name} - done. '
                    f'Train: {df_trx_train.count()}, test: {0}')

    def get_trx(self, path):
        spark = SparkSession.builder.getOrCreate()
        file_name = os.path.basename(path)

        df_trx = spark.read.parquet(path)
        df_trx = df_trx.withColumn('event_time', F.col(self.COL_EVENT_TIME))
        for col in df_trx.columns:
            df_trx = df_trx.withColumnRenamed(col, col.lower())
        for col in self.config.cols_category:
            df_trx = df_trx.withColumn(col, F.col(col) + 1)
        # df_trx = self.remove_long_trx(df_trx, self.config.max_trx_count, self.config.col_client_id)
        df_trx = self.collect_lists(df_trx, self.config.col_client_id)
        return df_trx, file_name

    def collect_lists(self, df, col_id):
        col_list = [col for col in df.columns if col != col_id]

        # if self.config.save_partitioned_data:
        #     df = df.withColumn('mon_id', (F.col('event_time') / 30).cast('int'))
        #     col_id = [col_id, 'mon_id']
        #
        df = df.withColumn('_rn', F.row_number().over(Window.partitionBy(col_id).orderBy('event_time')))

        df = df.groupby(col_id).agg(*[
            F.sort_array(F.collect_list(F.struct('_rn', col))).alias(col)
            for col in col_list
        ])
        for col in col_list:
            df = df.withColumn(col, F.col(f'{col}.{col}'))

        # df = df.drop('_rn')
        return df


if __name__ == '__main__':
    LocalDatasetConverter().run()
