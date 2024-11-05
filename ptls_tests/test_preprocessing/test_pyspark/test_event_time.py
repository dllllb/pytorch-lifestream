import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType

from ptls.preprocessing.pyspark.event_time import dt_to_timestamp, timestamp_to_dt, DatetimeToTimestamp


def test_dt_to_timestamp():
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    df = spark.createDataFrame(data=[
        {'dt': '1970-01-01 00:00:00'},
        {'dt': '2012-01-01 12:01:16'},
        {'dt': '2021-12-30 00:00:00'}
    ])

    df = df.withColumn('ts', dt_to_timestamp('dt'))
    ts = [rec.ts for rec in df.select('ts').collect()]
    spark.conf.unset("spark.sql.session.timeZone")
    assert ts == [0, 1325419276, 1640822400]



def test_timestamp_to_dt():
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    df = spark.createDataFrame(data=[
        {'dt': '1970-01-01 00:00:00'},
        {'dt': '2012-01-01 12:01:16'},
        {'dt': '2021-12-30 00:00:00'}
    ])
    from pyspark.sql import functions as F

    # df = df.withColumn('ts', dt_to_timestamp('dt'))
    # df = df.withColumn('dt2', timestamp_to_dt('ts'))

    from pyspark.sql.functions import col, unix_timestamp, from_unixtime

    df = df.withColumn("ts", unix_timestamp("dt"))
    df = df.withColumn("dt2", from_unixtime(col("ts")))

    df = df.toPandas()
    # df['dt'] = pd.to_datetime(df['dt'])  # Ensure 'dt' is datetime for comparison
    df['dt2'] = pd.to_datetime(df['dt2'])
    spark.conf.unset("spark.sql.session.timeZone")
    assert (df['dt'] == df['dt2']).all()
    assert df['dt'].dtype == 'object'
    assert df['ts'].dtype == 'int64'
    assert df['dt2'].dtype == 'datetime64[ns]'



def test_datetime_to_timestamp():
    t = DatetimeToTimestamp(col_name_original='dt')
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    df = spark.createDataFrame(data=[
        {'dt': '1970-01-01 00:00:00', 'rn': 1},
        {'dt': '2012-01-01 12:01:16', 'rn': 2},
        {'dt': '2021-12-30 00:00:00', 'rn': 3}
    ])
    df = t.fit_transform(df)
    et = [rec.event_time for rec in df.select('event_time').collect()]
    spark.conf.unset("spark.sql.session.timeZone")
    assert et[0] == 0
    assert et[1] == 1325419276
    assert et[2] == 1640822400