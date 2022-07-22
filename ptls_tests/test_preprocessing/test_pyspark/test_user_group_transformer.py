import numpy as np
from pyspark.sql import SparkSession

from ptls.preprocessing.pyspark.user_group_transformer import UserGroupTransformer


def test_group():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data=[
        {'user_id': 0, 'event_time': 0, 'mcc': 0, 'amount': 10.0},
        {'user_id': 0, 'event_time': 1, 'mcc': 1, 'amount': 11.0},
        {'user_id': 0, 'event_time': 2, 'mcc': 2, 'amount': 12.0},
        {'user_id': 1, 'event_time': 0, 'mcc': 3, 'amount': 13.0},
        {'user_id': 1, 'event_time': 1, 'mcc': 4, 'amount': 14.0},
        {'user_id': 1, 'event_time': 2, 'mcc': 5, 'amount': 15.0},
        {'user_id': 1, 'event_time': 3, 'mcc': 6, 'amount': 16.0},
        {'user_id': 2, 'event_time': 0, 'mcc': 7, 'amount': 17.0},
        {'user_id': 2, 'event_time': 1, 'mcc': 8, 'amount': 18.0},
    ])
    t = UserGroupTransformer(col_name_original='user_id')
    records = t.fit_transform(df).collect()
    rec = records[1]
    assert rec['user_id'] == 1
    np.testing.assert_equal(rec['event_time'], np.array([0, 1, 2, 3]))
    np.testing.assert_equal(rec['mcc'], np.array([3, 4, 5, 6]))
    np.testing.assert_equal(rec['amount'], np.array([13.0, 14.0, 15.0, 16.0]))


def test_group_with_target():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data=[
        {'user_id': 0, 'event_time': 0, 'mcc': 0, 'target': 10.0},
        {'user_id': 0, 'event_time': 1, 'mcc': 1, 'target': 11.0},
        {'user_id': 0, 'event_time': 2, 'mcc': 2, 'target': 12.0},
        {'user_id': 1, 'event_time': 0, 'mcc': 3, 'target': 13.0},
        {'user_id': 1, 'event_time': 1, 'mcc': 4, 'target': 14.0},
        {'user_id': 1, 'event_time': -1, 'mcc': 5, 'target': 15.0},
        {'user_id': 1, 'event_time': 3, 'mcc': 6, 'target': 16.0},
        {'user_id': 2, 'event_time': 0, 'mcc': 7, 'target': 17.0},
        {'user_id': 2, 'event_time': 1, 'mcc': 8, 'target': 18.0},
    ])
    t = UserGroupTransformer(col_name_original='user_id', cols_last_item=['target'])
    records = t.fit_transform(df).collect()
    rec = records[1]
    assert rec['user_id'] == 1
    np.testing.assert_equal(rec['event_time'], np.array([-1, 0, 1, 3]))
    np.testing.assert_equal(rec['mcc'], np.array([5, 3, 4, 6]))
    assert rec['target'] == 16


def test_trx_limit():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data=[
        {'user_id': 0, 'event_time': 0, 'mcc': 0, 'amount': 10.0},
        {'user_id': 0, 'event_time': 1, 'mcc': 1, 'amount': 11.0},
        {'user_id': 0, 'event_time': 2, 'mcc': 2, 'amount': 12.0},
        {'user_id': 1, 'event_time': 0, 'mcc': 3, 'amount': 13.0},
        {'user_id': 1, 'event_time': 1, 'mcc': 4, 'amount': 14.0},
        {'user_id': 1, 'event_time': 2, 'mcc': 5, 'amount': 15.0},
        {'user_id': 1, 'event_time': 3, 'mcc': 6, 'amount': 16.0},
        {'user_id': 2, 'event_time': 0, 'mcc': 7, 'amount': 17.0},
        {'user_id': 2, 'event_time': 1, 'mcc': 8, 'amount': 18.0},
    ])
    t = UserGroupTransformer(col_name_original='user_id', max_trx_count=3)
    records = t.fit_transform(df).collect()
    rec = records[1]
    assert rec['user_id'] == 1
    np.testing.assert_equal(rec['event_time'], np.array([1, 2, 3]))
    np.testing.assert_equal(rec['mcc'], np.array([4, 5, 6]))
    np.testing.assert_equal(rec['amount'], np.array([14.0, 15.0, 16.0]))
