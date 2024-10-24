from pyspark.sql import SparkSession
from pathlib import Path

from ptls.preprocessing.pyspark.pyspark_preprocessor import PysparkDataPreprocessor


def test_pandas_data_preprocessor():
    spark = SparkSession.builder.getOrCreate()
    data = spark.createDataFrame(data=[
        {'id': 1, 'event_time': 1, 'cat': 'a', 'amnt': 10, 'labels': 1000},
        {'id': 1, 'event_time': 2, 'cat': 'b', 'amnt': 11, 'labels': 1001},
        {'id': 1, 'event_time': 3, 'cat': 'c', 'amnt': 12, 'labels': 1002},
        {'id': 2, 'event_time': 4, 'cat': 'a', 'amnt': 13, 'labels': 1003},
        {'id': 2, 'event_time': 5, 'cat': 'b', 'amnt': 14, 'labels': 1004},
        {'id': 3, 'event_time': 6, 'cat': 'c', 'amnt': 15, 'labels': 1000},
    ])
    pp = PysparkDataPreprocessor(
        col_id='id',
        col_event_time='event_time',
        event_time_transformation='none',
        cols_category=['cat'],
        cols_numerical=['amnt'],
        cols_identity=['labels'],
    )
    features = pp.fit_transform(data)
    features = pp.transform(data).collect()
    assert len(features) == 3


def test_csv():
    spark = SparkSession.builder.getOrCreate()
    data_path = Path(__file__).parent / '..' / "age-transactions.csv"
    source_data = spark.read.csv(str(data_path), header=True)

    preprocessor = PysparkDataPreprocessor(
        col_id='client_id',
        col_event_time='trans_date',
        event_time_transformation='none',
        cols_category=["trans_date", "small_group"],
        cols_numerical=["amount_rur"],
    )

    dataset = preprocessor.fit_transform(source_data).collect()
    assert len(dataset) == 100
    assert len(dataset[0]) == 5
