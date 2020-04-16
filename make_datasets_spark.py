import argparse
import datetime
import logging
import os
import pickle
from random import Random

import numpy as np
import pandas as pd

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql import Window


logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=os.path.abspath)
    parser.add_argument('--trx_files', nargs='+')
    parser.add_argument('--target_files', nargs='*', default=[])

    parser.add_argument('--print_dataset_info', action='store_true')
    parser.add_argument('--col_client_id', type=str)
    parser.add_argument('--cols_event_time', nargs='+')
    parser.add_argument('--cols_category', nargs='*', default=[])
    parser.add_argument('--cols_log_norm', nargs='*', default=[])
    parser.add_argument('--col_target', required=False, type=str)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--salt', type=int, default=42)
    parser.add_argument('--max_trx_count', type=int, default=5000)

    parser.add_argument('--output_train_path', type=os.path.abspath)
    parser.add_argument('--output_test_path', type=os.path.abspath)
    parser.add_argument('--output_test_ids_path', type=os.path.abspath)
    parser.add_argument('--log_file', type=os.path.abspath)

    args = parser.parse_args(args)
    logger.info('Parsed args:\n' + '\n'.join([f'  {k:15}: {v}' for k, v in vars(args).items()]))
    return args


def load_source_data(data_path, trx_files):
    spark = SparkSession.builder.getOrCreate()

    data = None
    for file in trx_files:
        file_path = os.path.join(data_path, file)
        df = spark.read.csv(file_path, header=True)
        data = df if data is None else data.union(df)
        logger.info(f'Loaded {df.count()} rows from "{file_path}"')

    cnt = data.count()
    logger.info(f'Loaded {cnt} rows in total')

    cnt_in_partition = 100000
    data = data.repartition((cnt + cnt_in_partition - 1) // cnt_in_partition)
    return data


def pd_hist(df, name, bins=10):
    data = df.select(name).toPandas()[name]

    if data.dtype.kind == 'f':
        bins = np.linspace(data.min(), data.max(), bins + 1).round(1)
    elif np.percentile(data, 99) - data.min() > bins - 1:
        bins = np.linspace(data.min(), np.percentile(data, 99), bins).astype(int).tolist() + [int(data.max() + 1)]
    else:
        bins = np.arange(data.min(), data.max() + 2, 1).astype(int)
    df = pd.cut(data, bins, right=False).rename(name)
    df = df.to_frame().assign(cnt=1).groupby(name)[['cnt']].sum()
    df['% of total'] = df['cnt'] / df['cnt'].sum()
    return df


def get_encoder(df, col_name):
    df = df.withColumn(col_name, F.coalesce(F.col(col_name).cast('string'), F.lit('#EMPTY')))

    col_orig = '_orig_' + col_name
    df = df.withColumnRenamed(col_name, col_orig)

    df_encoder = df.groupby(col_orig).agg(F.count(F.lit(1)).alias('_cnt'))
    df_encoder = df_encoder.withColumn(col_name,
                                       F.row_number().over(Window.partitionBy().orderBy(F.col('_cnt').desc())))
    df_encoder = df_encoder.withColumn(col_name, F.col(col_name))
    df_encoder = df_encoder.drop('_cnt')

    df_encoder = df_encoder.repartition(1)
    df_encoder.persist()
    _ = df_encoder.count()

    return df_encoder


def encode_col(df, col_name, df_encoder):
    df = df.withColumn(col_name, F.coalesce(F.col(col_name).cast('string'), F.lit('#EMPTY')))

    col_orig = '_orig_' + col_name
    df = df.withColumnRenamed(col_name, col_orig)

    df = df.join(df_encoder, on=col_orig, how='left')
    df = df.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(1)))
    df = df.drop(col_orig)

    return df


def log_transform(df, col_name):
    df = df.withColumn(col_name, F.signum(F.col(col_name)) * F.log(F.abs(F.col(col_name)) + F.lit(1)))
    return df


def _td_default(df, cols_event_time):
    raise NotImplementedError()


def _td_float(df, col_event_time):
    df = df.withColumn('event_time', F.col(col_event_time).astype('float'))
    logger.info('To-float time transformation')
    return df


def _td_gender(df, col_event_time):
    """Gender-dataset-like transformation

    'd hh:mm:ss' -> float where integer part is day number and fractional part is seconds from day begin
    '1 00:00:00' -> 1.0
    '1 12:00:00' -> 1.5
    '1 01:00:00' -> 1 + 1 / 24
    '2 23:59:59' -> 1.99
    '432 12:00:00' -> 432.5

    :param df:
    :param col_event_time:
    :return:
    """
    df = df.withColumn('_et_day', F.substring(F.lpad(F.col(col_event_time), 15, '0'), 1, 6).cast('float'))

    df = df.withColumn('_et_time', F.substring(F.lpad(F.col(col_event_time), 15, '0'), 7, 9))
    df = df.withColumn('_et_time', F.regexp_replace('_et_time', r'\:60$', ':59'))
    df = df.withColumn('_et_time', F.unix_timestamp('_et_time', 'HH:mm:ss') / (24 * 60 * 60))

    df = df.withColumn('event_time', F.col('_et_day') + F.col('_et_time'))
    df = df.drop('_et_day', '_et_time')
    logger.info('Gender-dataset-like time transformation')
    return df


def remove_long_trx(df, max_trx_count, col_client_id):
    df = df.withColumn('_cn', F.count(F.lit(1)).over(Window.partitionBy(col_client_id)))
    df = df.withColumn('_rn', F.row_number().over(
        Window.partitionBy(col_client_id).orderBy(F.col('event_time').desc())))
    df = df.filter(F.col('_rn') <= max_trx_count)
    df = df.drop('_cn')
    df = df.drop('_rn')
    return df


def collect_lists(df, col_id):
    col_list = [col for col in df.columns if col != col_id]
    df = df \
        .withColumn('trx_count', F.count(F.lit(1)).over(Window.partitionBy(col_id))) \
        .withColumn('_rn', F.row_number().over(Window.partitionBy(col_id).orderBy('event_time')))

    for col in col_list:
        df = df.withColumn(col, F.collect_list(col).over(Window.partitionBy(col_id).orderBy('_rn'))) \

    df = df.filter('_rn = trx_count').drop('_rn')
    return df


def trx_to_features(df_data, print_dataset_info,
                    col_client_id, cols_event_time, cols_category, cols_log_norm, max_trx_count):
    if print_dataset_info:
        unique_clients = df_data.select(col_client_id).distinct().count()
        logger.info(f'Found {unique_clients} unique clients')

    # event_time mapping
    if cols_event_time[0][0] == '#':
        if cols_event_time[0] == '#float':
            df_data = _td_float(df_data, cols_event_time[1])
        elif cols_event_time[0] == '#gender':
            df_data = _td_gender(df_data, cols_event_time[1])
        else:
            raise NotImplementedError(f'Unknown type of data transformation: "{cols_event_time[0]}"')
    else:
        df_data = _td_default(df_data, cols_event_time)

    for col in cols_log_norm:
        df_data = log_transform(df_data, col)
        if print_dataset_info:
            logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{pd_hist(df_data, col)}')

    encoders = {col: get_encoder(df_data, col) for col in cols_category}
    for col in cols_category:
        df_data = encode_col(df_data, col, encoders[col])
        if print_dataset_info:
            logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{pd_hist(df_data, col)}')

    if print_dataset_info:
        df = df_data.groupby(col_client_id).agg(F.count(F.lit(1)).alias("trx_count"))
        logger.info(f'Trx count per clients:\nlen(trx_list) | client_count\n{pd_hist(df, "trx_count")}')

    # column filter
    used_columns = [col for col in df_data.columns
                    if col in cols_category + cols_log_norm + ['event_time', col_client_id]]

    logger.info('Feature collection in progress ...')
    features = df_data.select(used_columns)
    features = remove_long_trx(features, max_trx_count, col_client_id)
    features = collect_lists(features, col_client_id)

    if print_dataset_info:
        feature_names = list(features.columns)
        logger.info(f'Feature names: {feature_names}')

    features.persist()
    logger.info(f'Prepared features for {features.count()} clients')
    return features


def update_with_target(features, data_path, target_files, col_client_id, col_target):
    df_target = load_source_data(data_path, target_files)
    df_target = df_target.select(
        F.col(col_client_id).cast('int').alias(col_client_id),
        F.col(col_target).cast('int').alias('target'),
    )
    df_target = df_target.repartition(1)

    features = features.join(df_target, on=col_client_id, how='left')
    features.persist()

    logger.info(f'Target updated for {features.count()} clients')
    return features


def split_dataset(all_data, test_size, data_path, target_files, col_client_id, salt):
    spark = SparkSession.builder.getOrCreate()

    df_target = load_source_data(data_path, target_files)
    df_target = df_target.withColumn(col_client_id, F.col(col_client_id).cast('int'))
    s_clients = set(cl[0] for cl in df_target.select(col_client_id).distinct().collect())

    # shuffle client list
    s_all_data_clients = set(cl[0] for cl in all_data.select(col_client_id).distinct().collect())
    s_clients = (cl_id for cl_id in s_clients if cl_id in s_all_data_clients)
    s_clients = sorted(s_clients)
    s_clients = [cl_id for cl_id in s_clients]
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


def save_features(df_data, save_path):
    df_data.write.parquet(save_path, mode='overwrite')
    logger.info(f'Saved to: "{save_path}"')


if __name__ == '__main__':
    _start = datetime.datetime.now()
    config = parse_args()
    spark = SparkSession.builder.getOrCreate()

    if config.log_file is not None:
        handlers = [logging.StreamHandler(), logging.FileHandler(config.log_file, mode='w')]
    else:
        handlers = None
    logging.basicConfig(level=logging.INFO, format='%(funcName)-20s   : %(message)s',
                        handlers=handlers)

    # description
    spark.sparkContext.setLocalProperty('callSite.short', 'load_source_data')
    source_data = load_source_data(
        data_path=config.data_path,
        trx_files=config.trx_files,
    )
    source_data = source_data.withColumn(config.col_client_id, F.col(config.col_client_id).cast('int'))

    # description
    spark.sparkContext.setLocalProperty('callSite.short', 'trx_to_features')
    client_features = trx_to_features(
        df_data=source_data,
        print_dataset_info=config.print_dataset_info,
        col_client_id=config.col_client_id,
        cols_event_time=config.cols_event_time,
        cols_category=config.cols_category,
        cols_log_norm=config.cols_log_norm,
        max_trx_count=config.max_trx_count,
    )

    if len(config.target_files) > 0 and config.col_target is not None:
        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'update_with_target')
        client_features = update_with_target(
            features=client_features,
            data_path=config.data_path,
            target_files=config.target_files,
            col_client_id=config.col_client_id,
            col_target=config.col_target,
        )

    if config.test_size > 0:
        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'split_dataset')
        train, test = split_dataset(
            all_data=client_features,
            test_size=config.test_size,
            data_path=config.data_path,
            target_files=config.target_files,
            col_client_id=config.col_client_id,
            salt=config.salt,
        )
    else:
        train = client_features

    # description
    spark.sparkContext.setLocalProperty('callSite.short', 'save_features')
    save_features(
        df_data=train,
        save_path=config.output_train_path,
    )

    if config.test_size > 0:
        save_features(
            df_data=test,
            save_path=config.output_test_path,
        )
        test_ids = test.select(config.col_client_id).toPandas()
        test_ids.to_csv(config.output_test_ids_path, index=False)

    _duration = datetime.datetime.now() - _start
    logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')
