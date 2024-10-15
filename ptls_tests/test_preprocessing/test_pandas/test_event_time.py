import pandas as pd

from ptls.preprocessing.dask.dask_transformation.event_time import DatetimeToTimestamp
from ptls.preprocessing.util import dt_to_timestamp, timestamp_to_dt


def test_dt_to_timestamp():
    x = pd.Series([
        '1970-01-01 00:00:00',
        '2012-01-01 12:01:16',
        '2021-12-30 00:00:00',
    ])
    y = dt_to_timestamp(x)
    assert y[0] == 0
    assert y[1] == 1325419276
    assert y[2] == 1640822400


def test_timestamp_to_dt():
    x = pd.Series([
        '1970-01-01 00:00:00',
        '2012-01-01 12:01:16',
        '2021-12-30 00:00:00',
    ])
    y = timestamp_to_dt(dt_to_timestamp(x))
    assert (y == x).all()


def test_datetime_to_timestamp():
    t = DatetimeToTimestamp(col_name_original='dt')
    df = pd.DataFrame({
        'dt': ['1970-01-01 00:00:00', '2012-01-01 12:01:16', '2021-12-30 00:00:00'],
        'rn': [1, 2, 3],
    })
    df = t.fit_transform(df)
    et = df['event_time'].values
    assert et[0] == 0
    assert et[1] == 1325419276
    assert et[2] == 1640822400
