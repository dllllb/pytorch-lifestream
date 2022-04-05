import pandas as pd

from dltranz.data_preprocessing.pandas_preprocessor import PandasDataPreprocessor


def test_pandas_data_preprocessor():
    data = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 3],
        'event_time': [1, 2, 3, 4, 5, 6],
        'cat': ['a', 'b', 'c', 'a', 'b', 'c'],
        'amnt': [10, 11, 12, 13, 14, 15],
        'labels': [1000, 1001, 1002, 1003, 1004, 1005],
    })
    pp = PandasDataPreprocessor(
        col_id='id',
        cols_event_time='event_time',
        cols_category=['cat'],
        cols_log_norm=['amnt'],
        cols_identity=['labels'],
        time_transformation='none',
    )
    features = pp.fit_transform(data)
    features = pp.transform(data)
    assert len(features) == 3

