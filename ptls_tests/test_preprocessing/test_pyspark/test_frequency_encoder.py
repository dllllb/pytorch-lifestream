from pyspark.sql import SparkSession

from ptls.preprocessing.pyspark.frequency_encoder import FrequencyEncoder


def test_fit():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 4},
        {'uid': 0, 'cat': 5},
        {'uid': 0, 'cat': 5},
        {'uid': 1, 'cat': 5},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
    ])
    t = FrequencyEncoder(
        col_name_original='cat',
    )
    t.fit(df)
    assert t.mapping == {'2': 1, '5': 2, '4': 3}
    assert t.other_values_code == 4
    assert t.dictionary_size == 5


def test_fit_transform():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 4},
        {'uid': 0, 'cat': 5},
        {'uid': 0, 'cat': 5},
        {'uid': 1, 'cat': 5},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
    ])
    t = FrequencyEncoder(
        col_name_original='cat',
    )
    out = t.fit_transform(df)
    assert [rec['cat'] for rec in out.collect()] == [3, 2, 2, 2, 1, 1, 1, 1]


def test_new():
    spark = SparkSession.builder.getOrCreate()
    df_train = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 4},
        {'uid': 0, 'cat': 5},
        {'uid': 0, 'cat': 5},
        {'uid': 1, 'cat': 5},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
    ])
    df_test = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 4},
        {'uid': 0, 'cat': 5},
        {'uid': 0, 'cat': 5},
        {'uid': 1, 'cat': 5},
        {'uid': 1, 'cat': 3},
        {'uid': 1, 'cat': 3},
        {'uid': 1, 'cat': 3},
        {'uid': 1, 'cat': 3},
    ])
    t = FrequencyEncoder(
        col_name_original='cat',
    )
    t.fit(df_train)
    out = t.transform(df_test)
    assert [rec['cat'] for rec in out.collect()] == [3, 2, 2, 2, 4, 4, 4, 4]

def test_limit_dict():
    spark = SparkSession.builder.getOrCreate()
    df_train = spark.createDataFrame(data=[
        {'uid': 0, 'cat': 4},
        {'uid': 0, 'cat': 5},
        {'uid': 0, 'cat': 5},
        {'uid': 1, 'cat': 5},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
        {'uid': 1, 'cat': 2},
    ])
    t = FrequencyEncoder(
        col_name_original='cat',
        max_cat_num=1,
    )
    out = t.fit_transform(df_train)
    assert [rec['cat'] for rec in out.collect()] == [2, 2, 2, 2, 1, 1, 1, 1]
