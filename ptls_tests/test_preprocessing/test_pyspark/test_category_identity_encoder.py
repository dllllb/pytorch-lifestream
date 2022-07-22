import pytest
from pyspark.sql import SparkSession

from ptls.preprocessing.pyspark.category_identity_encoder import CategoryIdentityEncoder


def test_fit():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 1},
        {'uid': 0, 'cat': 2},
        {'uid': 0, 'cat': 3},
        {'uid': 1, 'cat': 4},
        {'uid': 1, 'cat': 1},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 3},
    ])
    t = CategoryIdentityEncoder(col_name_original='cat')
    t.fit(df)
    assert t.min_fit_index == 1
    assert t.max_fit_index == 4


def test_fit_transform():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 1},
        {'uid': 0, 'cat': 2},
        {'uid': 0, 'cat': 3},
        {'uid': 1, 'cat': 4},
        {'uid': 1, 'cat': 1},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 3},
    ])
    t = CategoryIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_new',
        is_drop_original_col=False,
    )
    out = t.fit_transform(df)
    assert out.toPandas()[['cat', 'cat_new']].diff(axis=1).iloc[:, 1].eq(0).all()


def test_fit_negative_error():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 1},
        {'uid': 0, 'cat': 2},
        {'uid': 0, 'cat': -3},
        {'uid': 1, 'cat': 4},
        {'uid': 1, 'cat': 1},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 3},
    ])
    t = CategoryIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_new',
        is_drop_original_col=False,
    )
    with pytest.raises(AttributeError):
        t.fit(df)


def test_fit_zero_warn():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 1},
        {'uid': 0, 'cat': 2},
        {'uid': 0, 'cat': 0},
        {'uid': 1, 'cat': 4},
        {'uid': 1, 'cat': 1},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 3},
    ])
    t = CategoryIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_new',
        is_drop_original_col=False,
    )
    with pytest.warns(UserWarning):
        out = t.fit_transform(df)

def test_fit_zero_warn():
    spark = SparkSession.builder.getOrCreate()
    df_train = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 1},
        {'uid': 0, 'cat': 2},
        {'uid': 0, 'cat': 1},
        {'uid': 1, 'cat': 4},
        {'uid': 1, 'cat': 1},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 3},
    ])
    df_test = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 1},
        {'uid': 0, 'cat': 2},
        {'uid': 0, 'cat': 5},
        {'uid': 1, 'cat': 4},
        {'uid': 1, 'cat': 1},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 3},
    ])
    t = CategoryIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_new',
        is_drop_original_col=False,
    )
    t.fit(df_train)
    with pytest.warns(UserWarning):
        out = t.transform(df_test)



