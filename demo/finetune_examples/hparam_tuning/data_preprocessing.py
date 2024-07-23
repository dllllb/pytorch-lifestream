import os
import pickle

import hydra
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window

from ptls.preprocessing import PysparkDataPreprocessor
import logging

logger = logging.getLogger(__name__)


def get_df_target(data_path):
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv(os.path.join(data_path, 'gender_train.csv'), header=True)
    df = df.withColumn('gender', F.col('gender').cast('int')).withColumnRenamed('gender', 'target_gender')
    logger.info(f'Loaded {df.count()} target records')
    return df


def split_target(df, salt, fold_count):
    df = df.withColumn(
        'hash', F.hash(F.concat(F.col('customer_id').cast('string'), F.lit(salt))) / 2 ** 32 + 0.5)
    df = df.withColumn(
        'fold_id', F.row_number().over(Window.partitionBy('target_gender').orderBy('hash')) % fold_count) \
        .drop('hash')
    return df


def get_df_trx(data_path):
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv(os.path.join(data_path, 'transactions.csv'), header=True)
    df = df.withColumn('amount', F.col('amount').cast('float'))
    df = df.withColumn('tr_datetime', F.lpad('tr_datetime', 15, '0'))
    df = df.withColumn('tr_datetime_d', F.substring('tr_datetime', 1, 6).cast('int') * (24 * 60 * 60))
    df = df.withColumn('tr_datetime_h', F.substring('tr_datetime', 8, 8))
    df = df.withColumn('tr_datetime_h', F.regexp_replace('tr_datetime_h', r'\:60$', ':59'))
    df = df.withColumn('tr_datetime_h', F.unix_timestamp('tr_datetime_h', 'HH:mm:ss').cast('int'))
    df = df.withColumn('event_time', (F.col('tr_datetime_d') + F.col('tr_datetime_h')))
    df = df.drop('tr_datetime', 'tr_datetime_d', 'tr_datetime_h', 'term_id')
    unique_user_cnt = df.select('customer_id').distinct().count()
    logger.info(f'Loaded {df.count()} transaction records with {unique_user_cnt} unique users')
    return df


def split_fold(fold_id, df_target, df_trx, conf_pp):
    spark = SparkSession.builder.getOrCreate()

    preproc = PysparkDataPreprocessor(
        col_id='customer_id',
        col_event_time='event_time',
        event_time_transformation='none',
        cols_category=['mcc_code', 'tr_type'],
        cols_numerical=['amount'],
        cols_identity=['term_id'],
        cols_last_item=['target_gender'],
    )
    df_train_trx = df_trx.join(df_target, how='left', on='customer_id').where(
        F.coalesce(F.col('fold_id'), F.lit(-1)) != fold_id).drop('fold_id')
    df_test_trx = df_trx.join(
        df_target.where(F.col('fold_id') == fold_id).drop('fold_id'),
        how='inner', on='customer_id')
    df_train_data = preproc.fit_transform(df_train_trx)
    df_test_data = preproc.transform(df_test_trx)
    file_name_train = get_file_name_train(fold_id)
    file_name_test = get_file_name_test(fold_id)
    df_train_data.write.parquet(os.path.join(conf_pp.folds_path, file_name_train), mode='overwrite')
    df_test_data.write.parquet(os.path.join(conf_pp.folds_path, file_name_test), mode='overwrite')
    with open(os.path.join(conf_pp.folds_path, f'preproc_{fold_id}.p'), 'wb') as f:
        pickle.dump(preproc, f)
    logger.info(f'Preprocessor[{fold_id}].category_dictionary_sizes={preproc.get_category_dictionary_sizes()}')

    for df_name in [file_name_train, file_name_test]:
        df = spark.read.parquet(os.path.join(conf_pp.folds_path, df_name))
        r_counts = df.groupby().agg(
            F.sum(F.when(F.col('target_gender').isNull(), F.lit(1)).otherwise(F.lit(0))).alias('cnt_none'),
            F.sum(F.when(F.col('target_gender') == 0, F.lit(1))).alias('cnt_0'),
            F.sum(F.when(F.col('target_gender') == 1, F.lit(1))).alias('cnt_1'),
        ).collect()[0]
        cnt_str = ', '.join([
            f'{r_counts.cnt_none:5d} unlabeled'
            f'{r_counts.cnt_0:5d} - 0 class'
            f'{r_counts.cnt_1:5d} - 1 class'
        ])
        logger.info(f'{df_name:30} {cnt_str}')

    for df_name in [file_name_train, file_name_test]:
        df = spark.read.parquet(os.path.join(conf_pp.folds_path, df_name))
        logger.info(f'{df_name:30} {df}')


def get_file_name_train(fold_id):
    file_name_train = f'df_train_data_{fold_id}.parquet'
    return file_name_train


def get_file_name_test(fold_id):
    file_name_test = f'df_test_data_{fold_id}.parquet'
    return file_name_test


@hydra.main(version_base=None, config_path="conf", config_name="data_preprocessing")
def main(conf):
    logger.info('Start')

    conf_pp = conf.data_preprocessing

    df_target = get_df_target(conf_pp.data_path)
    fold_count = conf_pp.fold_count
    df_target = split_target(df_target, salt=conf_pp.salt, fold_count=fold_count)

    df_trx = get_df_trx(conf_pp.data_path)

    df_target.persist()

    for fold_id in range(fold_count):
        split_fold(fold_id, df_target, df_trx, conf_pp)
    logger.info('Done')


if __name__ == '__main__':
    main()
