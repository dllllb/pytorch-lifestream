import argparse
import datetime
import logging
import os
import pickle
from random import Random
from typing import List, Optional

import numpy as np
import pandas as pd

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import DataFrame  # For typing


logger = logging.getLogger(__name__)


class DatasetConverter:
    """
    Converts datasets from transaction list to features for metric learning.

    The class is designed to be run from command line with arguments.
    Call python3 make_datasets_spark.py --help to see arguments description.
    """
    def __init__(self):
        self.config = None

    def parse_args(self, args: Optional[List[str]]=None) -> None:
        """
        Parses command line arguments and saves them to self.config.

        Arguments:
        ----------
        args: Optional[List[str]]
            List of arguments to parse. If None, sys.argv is used.
        """
        parser = argparse.ArgumentParser()

        parser.add_argument('--data_path', type=os.path.abspath, 
                            help='Path to the directory containing trx files (datasets)')
        parser.add_argument('--trx_files', nargs='+',
                            help='List of dataset filenames with transaction features. ' \
                                 'Note: target column will be ignored.' \
                                 'Please use --target_files to specify targets.')
        parser.add_argument('--target_files', nargs='*', default=[],
                            help='List of target files containing client_id and target columns. '\
                                'The files can overlap with trx_files or be separate.')
        parser.add_argument('--target_as_array', action='store_true')

        parser.add_argument('--print_dataset_info', action='store_true')
        parser.add_argument('--sample_fraction', type=float, default=None)
        parser.add_argument('--col_client_id', type=str)
        parser.add_argument('--cols_event_time', nargs='+', 
                            help='Two arguments: 1) type of time transformation ' \
                                 '2) time column name.\n' \
                                 'Possible time transformation types: ' \
                                 '"#float", "#datetime", "#gender"')

        parser.add_argument('--dict', nargs='*', default=[])
        parser.add_argument('--cols_category', nargs='*', default=[],
                            help = 'List of categorical columns. All categorical ' \
                                   'features are encoded with embedding indexes. ' \
                                   'The indexes correspond to frequency rank:' \
                                   'All values are sorted by frequency in descending order ' \
                                   'and are numbered according to the order. ' \
                                   'The most common value will be replaced with 1, ' \
                                   'second common value will be replaced with 2 etc.')
        parser.add_argument('--cols_log_norm', nargs='*', default=[],
                            help='List of columns to apply log transformation to. ' \
                                 'Log transformation is applied as signum(x) * log(|x| + 1)')
        parser.add_argument('--col_target', nargs='*', default=[])
        parser.add_argument('--test_size', default='0.1')
        parser.add_argument('--salt', type=int, default=42,
                            help='Random seed for client shuffling')
        parser.add_argument('--max_trx_count', type=int, default=5000,
                            help='All sequences (transactions) ' \
                                 'exceeding this number will be removed')

        parser.add_argument('--output_train_path', type=os.path.abspath)
        parser.add_argument('--output_test_path', type=os.path.abspath)
        parser.add_argument('--output_test_ids_path', type=os.path.abspath)
        parser.add_argument('--save_partitioned_data', action='store_true')
        parser.add_argument('--log_file', type=os.path.abspath, 
                            help='File to dump logs to. If set logs will ' \
                            'be present in stdout and in the file, otherwise only in stdout. ' \
                            'Notice that stdout will always contain both ' \
                            'Spark logs and script logs, which makes it hard to read. ' \
                            'Thus, log_file is useful to be able to read only script logs.')


        args = parser.parse_args(args)
        logger.info('Parsed args:\n' + '\n'.join([f'  {k:15}: {v}' for k, v in vars(args).items()]))
        self.config = args

    def spark_read_file(self, path: str):
        """
        Creates a spark.DataFrame from a given file 
        using the file extension to determine the format.
        """
        spark = SparkSession.builder.getOrCreate()

        ext = os.path.splitext(path)[1]
        if ext == '.csv':
            return spark.read.option("escape", "\"").csv(path, header=True)
        elif ext == '.parquet':
            return spark.read.parquet(path)
        else:
            raise AttributeError(f'Unknown extension "{ext}" for "{path}"')

    def path_to_file(self, file_name):
        return os.path.join(self.config.data_path, file_name)

    def load_source_data(self, trx_files: List[str]):
        """
        Arguments:
        ----------
        trx_files: List[str]
            List of filenames stored in `self.config.data_path` 
            directory to load data from.

        Returns:
        --------
        data: spark.DataFrame
            spark.DataFrame with `event_time` column of float type
        """
        data = None
        for file in trx_files:
            file_path = self.path_to_file(file)
            df = self.spark_read_file(file_path)
            data = df if data is None else data.union(df)
            logger.info(f'Loaded {df.count()} rows from "{file_path}"')

        cnt = data.count()
        logger.info(f'Loaded {cnt} rows in total')

        cnt_in_partition = 100000
        data = data.repartition((cnt + cnt_in_partition - 1) // cnt_in_partition)
        return data

    def pd_hist(self, df, name, bins=10):
        # logger.info('pd_hist begin')
        # logger.info(f'sf = {self.config.sample_fraction}')
        data = df.select(name)
        if self.config.sample_fraction is not None:
            data = data.sample(fraction=self.config.sample_fraction)
        data = data.toPandas()[name]

        if data.dtype.kind == 'f':
            round_len = 1 if data.max() > bins + 1 else 2
            bins = np.linspace(data.min(), data.max(), bins + 1).round(round_len)
        elif np.percentile(data, 99) - data.min() > bins - 1:
            bins = np.linspace(data.min(), np.percentile(data, 99), bins).astype(int).tolist() + [int(data.max() + 1)]
        else:
            bins = np.arange(data.min(), data.max() + 2, 1).astype(int)
        df = pd.cut(data, bins, right=False).rename(name)
        df = df.to_frame().assign(cnt=1).groupby(name)[['cnt']].sum()
        df['% of total'] = df['cnt'] / df['cnt'].sum()
        return df

    def get_encoder(self, df: DataFrame, col_name: str) -> DataFrame:
        df = df.withColumn(col_name, F.coalesce(F.col(col_name).cast('string'), F.lit('#EMPTY')))

        col_orig = '_orig_' + col_name
        df = df.withColumnRenamed(col_name, col_orig)

        df_encoder = df.groupby(col_orig).agg(F.count(F.lit(1)).alias('_cnt'))
        df_encoder = df_encoder.withColumn(col_name,
                                           F.row_number().over(Window.partitionBy().orderBy(F.col('_cnt').desc())))
        df_encoder = df_encoder.withColumn(col_name, F.col(col_name) + F.lit(1))
        df_encoder = df_encoder.withColumn(col_name, F.col(col_name))
        df_encoder = df_encoder.drop('_cnt')

        df_encoder = df_encoder.repartition(1)
        df_encoder.persist()

        # AFAIU this call is to trigger the computation since pyspark is lazy.
        _ = df_encoder.count()

        return df_encoder

    def encode_col(self, df, col_name, df_encoder):
        df = df.withColumn(col_name, F.coalesce(F.col(col_name).cast('string'), F.lit('#EMPTY')))

        col_orig = '_orig_' + col_name
        df = df.withColumnRenamed(col_name, col_orig)

        df = df.join(df_encoder, on=col_orig, how='left')
        df = df.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(1)))
        df = df.drop(col_orig)

        return df

    def log_transform(self, df, col_name):
        df = df.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0)))
        df = df.withColumn(col_name, F.signum(F.col(col_name)) * F.log(F.abs(F.col(col_name)) + F.lit(1)))
        return df

    def _td_default(self, df, cols_event_time):
        raise NotImplementedError()

    def _td_float(self, df, col_event_time):
        df = df.withColumn('event_time', F.col(col_event_time).astype('float'))
        logger.info('To-float time transformation')
        return df

    def _td_datetime(self, df, col_event_time):
        df = df.withColumn('event_time', F.unix_timestamp(F.col(col_event_time)) / F.lit(24 * 60 * 60))
        logger.info('Datetime-to-unix-timestamp time transformation')
        return df

    def _td_gender(self, df, col_event_time):
        """Gender-dataset-like transformation

        'd hh:mm:ss' -> float where integer part is day number and fractional part is seconds from day begin
        '1 00:00:00' -> 1.0
        '1 12:00:00' -> 1.5
        '1 01:00:00' -> 1 + 1 / 24
        '2 23:59:59' -> 1.99
        '432 12:00:00' -> 432.5   '000432 12:00:00'

        :param df:
        :param col_event_time:
        :return:
        """
        df = df.withColumn('_et_day', F.substring(F.lpad(F.col(col_event_time), 15, '0'), 1, 6).cast('float'))

        df = df.withColumn('_et_time', F.substring(F.lpad(F.col(col_event_time), 15, '0'), 8, 8))
        df = df.withColumn('_et_time', F.regexp_replace('_et_time', r'\:60$', ':59'))
        df = df.withColumn('_et_time', F.unix_timestamp('_et_time', 'HH:mm:ss') / (24 * 60 * 60))

        df = df.withColumn('event_time', F.col('_et_day') + F.col('_et_time'))
        df = df.drop('_et_day', '_et_time')
        logger.info('Gender-dataset-like time transformation')
        return df

    def remove_long_trx(self, df, max_trx_count, col_client_id):
        """
        This function select the last max_trx_count transactions
        """
        df = df.withColumn('_cn', F.count(F.lit(1)).over(Window.partitionBy(col_client_id)))
        df = df.withColumn('_rn', F.row_number().over(
            Window.partitionBy(col_client_id).orderBy(F.col('event_time').desc())))
        df = df.filter(F.col('_rn') <= max_trx_count)
        df = df.drop('_cn')
        df = df.drop('_rn')
        return df

    def collect_lists(self, df, col_id):
        col_list = ['event_time'] + [col for col in df.columns if col != col_id and col != 'event_time']
        unpack_col_list = [col_id] + [F.col(f'_struct.{col}').alias(col) for col in col_list]

        if self.config.save_partitioned_data:
            df = df.withColumn('mon_id', (F.col('event_time') / 30).cast('int'))
            col_id = [col_id, 'mon_id']
            unpack_col_list.append('mon_id')

        # Put columns into structs and collect structs.
        df = df.groupBy(col_id).agg(F.sort_array(F.collect_list(F.struct(*col_list))).alias('_struct'))
        # Unpack structs.
        df = df.select(*unpack_col_list).drop('_struct').persist()
        # Get counts.
        df = df.withColumn('trx_count', F.size(F.col('event_time')).cast('long'))
        return df

    def join_dict(self, df, df_dict_name, col_id):
        path = self.path_to_file(df_dict_name)
        df_dict = self.spark_read_file(path)
        df = df.join(df_dict, on=col_id, how='left')

        col_counter = 0
        for col in df_dict.columns:
            if col == col_id:
                continue
            col_counter += 1
        logger.info(f'Join with "{path}" done. New {col_counter} columns joined')
        return df

    def trx_to_features(self, df_data, print_dataset_info: bool,
                        col_client_id, cols_event_time, cols_category, cols_log_norm, max_trx_count: int):
        if print_dataset_info:
            unique_clients = df_data.select(col_client_id).distinct().count()
            logger.info(f'Found {unique_clients} unique clients')

        for col in cols_log_norm:
            df_data = self.log_transform(df_data, col)
            if print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{self.pd_hist(df_data, col)}')

        encoders = {col: self.get_encoder(df_data, col) for col in cols_category}
        for col in cols_category:
            df_data = self.encode_col(df_data, col, encoders[col])
            if print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{self.pd_hist(df_data, col)}')

        if print_dataset_info:
            df = df_data.groupby(col_client_id).agg(F.count(F.lit(1)).alias("trx_count"))
            logger.info(f'Trx count per clients:\nlen(trx_list) | client_count\n{self.pd_hist(df, "trx_count")}')

        # column filter
        used_columns = [col for col in df_data.columns
                        if col in cols_category + cols_log_norm + ['event_time', col_client_id]]

        logger.info('Feature collection in progress ...')
        features = df_data.select(used_columns)
        features = self.remove_long_trx(features, max_trx_count, col_client_id)
        features = self.collect_lists(features, col_client_id)

        if print_dataset_info:
            feature_names = list(features.columns)
            logger.info(f'Feature names: {feature_names}')

        features.persist()
        logger.info(f'Prepared features for {features.count()} clients')
        return features

    def update_with_target(self, features, df_target, col_client_id, col_target):
        if type(col_target) is list and self.config.target_as_array:
            col_list = []
            for col in col_target:
                col_list.append(F.col(col))
            df_target = df_target.withColumn("target", F.array(col_list)) 
            df_target = df_target.select(col_client_id, "target")
        elif type(col_target) is list and not self.config.target_as_array:
            col_list = []
            for col in col_target:
                if col.startswith('target'):
                    col_list.append(F.col(col))
                else:
                    col_list.append(F.col(col).alias(f'target_{col}'))
            df_target = df_target.select(col_client_id, *col_list)
        else:
            col_list = [F.col(col_client_id).alias(col_client_id)]
            col_list.append(F.col(col_target).cast('int').alias('target'))
            df_target = df_target.select(*col_list)
            df_target = df_target.repartition(1)

        features = features.join(df_target, on=col_client_id, how='left')
        features.persist()

        logger.info(f'Target updated for {features.count()} clients')
        return features

    def split_dataset(self, all_data, test_size, df_target, col_client_id, salt):
        spark = SparkSession.builder.getOrCreate()

        s_clients = set(cl[0] for cl in df_target.select(col_client_id).distinct().collect())

        # shuffle client list
        s_all_data_clients = set(cl[0] for cl in all_data.select(col_client_id).distinct().collect())
        s_clients = sorted(cl_id for cl_id in s_clients if cl_id in s_all_data_clients)
        Random(salt).shuffle(s_clients)

        # split client list
        Nrows_test = int(len(s_clients) * test_size)
        s_clients_train = s_clients[:-Nrows_test]
        s_clients_test = s_clients[-Nrows_test:]

        s_clients_train = spark.createDataFrame([(i,) for i in s_clients_train], [col_client_id]).repartition(1)
        s_clients_test = spark.createDataFrame([(i,) for i in s_clients_test], [col_client_id]).repartition(1)
        s_clients = spark.createDataFrame([(i,) for i in s_clients], [col_client_id]).repartition(1)

        # split data
        labeled_train = all_data.join(s_clients_train, on=col_client_id, how='inner')
        labeled_test = all_data.join(s_clients_test, on=col_client_id, how='inner')
        unlabeled = all_data.join(s_clients, on=col_client_id, how='left_anti')

        train = labeled_train.union(unlabeled)
        test = labeled_test

        logger.info(f'Train size: {train.count()} clients')
        logger.info(f'Test size: {test.count()} clients')

        return train, test

    def split_dataset_predefined(
                self,
                all_data,
                data_path,
                col_client_id,
                test_ids_path,
            ):
        df_test = self.load_source_data([test_ids_path])
        df_test = df_test.withColumn('_is_test', F.lit(1))

        all_data = all_data.join(df_test, on=col_client_id, how='left')
        all_data = all_data.withColumn('_is_test', F.coalesce(F.col('_is_test'), F.lit(0)))

        train = all_data.filter("_is_test = 0")
        test = all_data.filter("_is_test = 1")
        return train, test

    def save_features(self, df_data, save_path):
        if not self.config.save_partitioned_data:
            df_data.write.parquet(save_path, mode='overwrite')
            logger.info(f'Saved to: "{save_path}"')
        else:
            df_data = df_data.withColumn('hash_id', F.crc32(F.col(self.config.col_client_id)) % 100)
            df_data.write.parquet(save_path, mode='overwrite', partitionBy=['mon_id', 'hash_id'])
            logger.info(f'Saved partitions to: "{save_path}"')

    def run(self):
        _start = datetime.datetime.now()
        self.parse_args()
        spark = SparkSession.builder.getOrCreate()

        self.logging_config()

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'load_source_data')
        source_data = self.load_transactions()

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'trx_to_features')
        client_features = self.trx_to_features(
            df_data=source_data,
            print_dataset_info=self.config.print_dataset_info,
            col_client_id=self.config.col_client_id,
            cols_event_time=self.config.cols_event_time,
            cols_category=self.config.cols_category,
            cols_log_norm=self.config.cols_log_norm,
            max_trx_count=self.config.max_trx_count,
        )

        if len(self.config.target_files) > 0 and len(self.config.col_target) > 0:
            # load target
            df_target = self.load_target()
            df_target.persist()

            if len(self.config.col_target) == 1:
                col_target = self.config.col_target[0]
            else:
                col_target = self.config.col_target

            # description
            spark.sparkContext.setLocalProperty('callSite.short', 'update_with_target')
            client_features = self.update_with_target(
                features=client_features,
                df_target=df_target,
                col_client_id=self.config.col_client_id,
                col_target=col_target,
            )

        train, test, save_test_id = None, None, False
        if self.config.test_size == 'predefined':
            train, test = self.split_dataset_predefined(
                all_data=client_features,
                data_path=self.config.data_path,
                col_client_id=self.config.col_client_id,
                test_ids_path=self.config.output_test_ids_path,
            )
        elif float(self.config.test_size) > 0:
            # description
            spark.sparkContext.setLocalProperty('callSite.short', 'split_dataset')
            train, test = self.split_dataset(
                all_data=client_features,
                test_size=float(self.config.test_size),
                df_target=df_target,
                col_client_id=self.config.col_client_id,
                salt=self.config.salt,
            )
            save_test_id = True
        else:
            train = client_features

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'save_features')
        self.save_features(
            df_data=train,
            save_path=self.config.output_train_path,
        )

        if test is not None:
            self.save_features(
                df_data=test,
                save_path=self.config.output_test_path,
            )

        if save_test_id:
            self.save_test_ids(test)

        _duration = datetime.datetime.now() - _start
        logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')

    def save_test_ids(self, df_test):
        test_ids = df_test.select(self.config.col_client_id).distinct().toPandas()
        test_ids.to_csv(self.config.output_test_ids_path, index=False)

    def load_target(self):
        df_target = self.load_source_data(self.config.target_files)
        return df_target

    def logging_config(self):
        if self.config.log_file is not None:
            handlers = [logging.StreamHandler(), logging.FileHandler(self.config.log_file, mode='w')]
        else:
            handlers = None
        logging.basicConfig(level=logging.INFO, format='%(funcName)-20s   : %(message)s', handlers=handlers)

    def load_transactions(self):
        """
        Returns a single spark.DataFrame with transaction 
        data collected from all trx_files.
        """
        spark = SparkSession.builder.getOrCreate()

        source_data = self.load_source_data(trx_files=self.config.trx_files)

        if len(self.config.dict) > 0:
            if len(self.config.dict) % 2 != 0:
                raise AttributeError('--dict should be in format (file_name col_id)*')
            for i in range(len(self.config.dict) // 2):
                # description
                spark.sparkContext.setLocalProperty('callSite.short', f'join with {i}th dict')
                source_data = self.join_dict(source_data, self.config.dict[2 * i], self.config.dict[2 * i + 1])

        # event_time mapping
        cols_event_time = self.config.cols_event_time
        if cols_event_time[0][0] == '#':
            if cols_event_time[0] == '#float':
                source_data = self._td_float(source_data, cols_event_time[1])
            elif cols_event_time[0] == '#datetime':
                source_data = self._td_datetime(source_data, cols_event_time[1])
            elif cols_event_time[0] == '#gender':
                source_data = self._td_gender(source_data, cols_event_time[1])
            else:
                raise NotImplementedError(f'Unknown type of data transformation: "{cols_event_time[0]}"')
        else:
            source_data = self._td_default(source_data, cols_event_time)

        return source_data


if __name__ == '__main__':
    DatasetConverter().run()
