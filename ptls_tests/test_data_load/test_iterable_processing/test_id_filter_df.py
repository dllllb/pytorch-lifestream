import numpy as np
import pandas as pd

from ptls.data_load.iterable_processing.id_filter_df import IdFilterDf


def test_df_filter_one_col():
    i_filter = IdFilterDf(pd.DataFrame({
        'col1': [8, 9, 1, 2, 3],
    }))

    data = [{'col1': i, 'feature_1': np.random.randint(1, 10, 100)} for i in range(20)]
    data = i_filter(data)
    data = [x['col1'] for x in data]
    assert data == [1, 2, 3, 8, 9]


def test_df_filter_two_col():
    i_filter = IdFilterDf(pd.DataFrame({
        'col1': [8, 9, 1, 2, 3],
        'col2': [1, 1, 1, 2, 2],
    }))

    data = [{'col1': i, 'col2': 1, 'feature_1': np.random.randint(1, 10, 100)} for i in range(20)]
    data = i_filter(data)
    data = [x['col1'] for x in data]
    assert data == [1, 8, 9]


def test_df_filter_two_col_type_cast_to_int():
    i_filter = IdFilterDf(pd.DataFrame({
        'col1': [8, 9, 1, 2, 3],
        'col2': [1, 1, 1, 2, 2],
    }))

    data = [{'col1': i, 'col2': '1', 'feature_1': np.random.randint(1, 10, 100)} for i in range(20)]
    data = i_filter(data)
    data = [x['col1'] for x in data]
    assert data == [1, 8, 9]


def test_df_filter_two_col_type_cast_to_datetime():
    i_filter = IdFilterDf(pd.DataFrame({
        'col1': [8, 9, 1, 2, 3],
        'col2': pd.to_datetime([
            '2012-01-08 00:00:00',
            '2012-01-08 00:00:00',
            '2012-01-08 00:00:00',
            '2020-01-08 00:00:00',
            '2020-01-08 00:00:00',
        ]),
    }))

    data = [{'col1': i, 'col2': '2012-01-08', 'feature_1': np.random.randint(1, 10, 100)} for i in range(20)]
    data = i_filter(data)
    data = [x['col1'] for x in data]
    assert data == [1, 8, 9]


def test_df_filter_no_relevant():
    i_filter = IdFilterDf(pd.DataFrame({
        'col1': [8, 9, 1, 2, 3],
        'col2': pd.to_datetime([
            '2012-01-08 00:00:00',
            '2012-02-08 00:00:00',
            '2012-03-08 00:00:00',
            '2012-04-08 00:00:00',
            '2012-05-08 00:00:00',
        ]),
    }))

    data = [{'col1': i, 'col2': '2012-01-03', 'feature_1': np.random.randint(1, 10, 100)} for i in range(20)]
    data = i_filter(data)
    data = [x['col1'] for x in data]
    assert data == []
