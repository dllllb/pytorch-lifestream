import pandas as pd
from pathlib import Path

from ptls.preprocessing.pandas.pandas_preprocessor import PandasDataPreprocessor


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
        col_event_time='event_time',
        event_time_transformation='none',
        cols_category=['cat'],
        cols_numerical=['amnt'],
        cols_identity=['labels'],
    )
    features = pp.fit_transform(data)
    features = pp.transform(data)
    assert len(features) == 3


def test_csv():
    data_path = Path(__file__).parent / '..' / "age-transactions.csv"
    source_data = pd.read_csv(data_path)

    source_data = source_data.rename({'trans_date': 'event_time'}, axis=1)

    preprocessor = PandasDataPreprocessor(
        col_id='client_id',
        col_event_time='event_time',
        event_time_transformation='none',
        cols_category=["event_time", "small_group"],
        cols_numerical=["amount_rur"],
        return_records=False,
    )

    dataset = preprocessor.fit_transform(source_data)
    # dataset = dataset.to_dict(orient='records')
    # dataset = preprocessor.fit_transform(source_data).to_dict(orient='records')
    assert len(dataset) == 100
    # assert len(dataset[0].keys()) == 5
