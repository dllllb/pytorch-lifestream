from make_datasets_spark import DatasetConverter

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
        spark = SparkSession.builder.getOrCreate()
        df_trx = spark.read.csv(self.path_to_file('transactions.csv.gz'), header=True)
        # df_trx = df_trx.limit(10000)
        df_trx = df_trx.select(
            F.col('id'),
            # F.col('chain').cast('int'),
            (F.unix_timestamp(F.col('date').cast('date')) / (24 * 60 * 60)).alias('event_time'),
            F.col('dept').cast('int'),
            F.col('category'),  # to be encoded
            F.col('productmeasure'),  # to be encoded
            F.concat(F.col('productmeasure'), F.col('productsize')).alias('productsize'),  # to be encoded
            F.col('purchaseamount').cast('float'),
        )
        df_trx = self.log_transform(df_trx, 'purchaseamount')

        df_trx = df_trx.repartition(500)
        df_trx.persist()
        logger.info(f'Loaded {df_trx.count()} transactions')

        return df_trx

    def load_target(self):
        raise NotImplementedError()

    def load_train_history(self):
        spark = SparkSession.builder.getOrCreate()
        df_train_hist = spark.read.csv('./data/trainHistory.csv.gz', header=True)

        df_train_hist = df_train_hist.select(
            F.col('id'),
            # F.col('chain').cast('int'),
            F.col('offer'),
            F.col('market').cast('int'),
            F.when(F.col('repeater') == 'f', F.lit(0)).otherwise(F.lit(1)).alias('repeater'),
        ).join(self.load_offers(), on='offer', how='inner')
        df_train_hist.persist()
        df_train_hist.count()
        return df_train_hist

    def load_test_history(self):
        spark = SparkSession.builder.getOrCreate()
        df_test_hist = spark.read.csv('./data/testHistory.csv.gz', header=True)

        df_test_hist = df_test_hist.select(
            F.col('id'),
            # F.col('chain').cast('int'),
            F.col('offer'),
            F.col('market').cast('int'),
            F.lit(-1).alias('repeater'),
        ).join(self.load_offers(), on='offer', how='inner')
        df_test_hist.persist()
        df_test_hist.count()
        return df_test_hist

    def load_offers(self):
        spark = SparkSession.builder.getOrCreate()
        df_offers = spark.read.csv('./data/offers.csv.gz', header=True)
        df_offers = df_offers.select(
            F.col('offer'),
            F.row_number().over(Window.partitionBy().orderBy(F.lit(1))).alias('offer_id'),
        )
        return df_offers

    def run(self):
        _start = datetime.datetime.now()
        self.parse_args()
        self.logging_config()

        spark = SparkSession.builder.getOrCreate()

        logger.info(f'Loading ...')
        df_train_hist = self.load_train_history()
        df_trx = self.load_transactions()

        df = df_train_hist
        ws = Window.partitionBy('repeater', 'offer', 'market')
        df = df.withColumn('_hash', F.hash(
            F.concat(F.col(self.config.col_client_id), F.lit(self.config.salt))) / 2**32 + 0.5)
        df = df.withColumn('p', F.row_number().over(ws.orderBy('_hash')) / F.count('*').over(ws))

        df_target_train = df.where(F.col('p') >= self.config.test_size).drop('_hash', 'p')
        df_target_test = df.where(F.col('p') < self.config.test_size).drop('_hash', 'p')
        logger.info(f'{df_target_train.count()} ids in train, {df_target_test.count()} in test')

        self.save_test_ids(df_target_test)

        logger.info(f'Start processing')
        df_trx_train = df_trx.join(
            df_target_test.select(self.config.col_client_id),
            on=self.config.col_client_id,
            how='left_anti',
        )
        cols_category = ['category', 'productmeasure', 'productsize']
        encoders = {col: self.get_encoder(df_trx_train, col)
                    for col in cols_category}
        encoders['productsize'] = encoders['productsize'].where("productsize < 99")  # top 99 sizes
        for col in cols_category:
            df_trx = self.encode_col(df_trx, col, encoders[col])
            logger.info(f'Encoded "{col}": {encoders[col].count()} items in dictionary')

        df_trx = self.remove_long_trx(df_trx, self.config.max_trx_count, self.config.col_client_id)
        df_trx = self.collect_lists(df_trx, self.config.col_client_id)

        df_trx_train = df_trx.join(
            df_target_test.select(self.config.col_client_id),
            on=self.config.col_client_id,
            how='left_anti',
        ).join(
            df_target_train.select(self.config.col_client_id, 'offer_id', 'market', 'repeater'),
            on=self.config.col_client_id,
            how='left',
        )
        df_trx_test = df_trx.join(
            df_target_test.select(self.config.col_client_id, 'offer_id', 'market', 'repeater'),
            on=self.config.col_client_id,
            how='inner',
        )

        train_path_to = self.config.output_train_path
        test_path_to = self.config.output_test_path
        df_trx_train.write.parquet(train_path_to, mode='overwrite')
        df_trx_test.write.parquet(test_path_to, mode='overwrite')

        df = spark.read.parquet(train_path_to)
        logger.info(f'Train size is {df.count()} unique clients')
        logger.info(f'Train column list: {df.columns}')

        df = spark.read.parquet(test_path_to)
        logger.info(f'Test size is {df.count()} unique clients')
        logger.info(f'Test column list: {df.columns}')

        _duration = datetime.datetime.now() - _start
        logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')

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
