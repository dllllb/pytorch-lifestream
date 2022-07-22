from pyspark.sql import SparkSession

from ptls.preprocessing.pyspark.col_identity_transformer import ColIdentityEncoder


def test_identity():
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
    t = ColIdentityEncoder(
        col_name_original='cat',
    )
    out = t.fit_transform(df)
    assert df.subtract(out).unionAll(out.subtract(df)).count() == 0


def test_new_col():
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
    t = ColIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_2',
        is_drop_original_col=False,
    )
    out = t.fit_transform(df)
    assert (df.toPandas()['cat'] == out.toPandas()['cat_2']).all()
    assert (df.toPandas()['cat'] == out.toPandas()['cat']).all()


def test_rename():
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
    t = ColIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_2',
        is_drop_original_col=True,
    )
    out = t.fit_transform(df)
    assert (df.toPandas()['cat'] == out.toPandas()['cat_2']).all()
    assert 'cat' not in out.columns
