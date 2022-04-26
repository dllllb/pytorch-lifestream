from pathlib import Path
import pandas as pd
from dltranz.data_preprocessing import PandasDataPreprocessor

def test_csv():
    data_path = Path(__file__).parent / "age-transactions.csv"
    source_data = pd.read_csv(data_path)

    preprocessor = PandasDataPreprocessor(
        col_id='client_id',
        cols_event_time='trans_date',
        time_transformation='float',
        cols_category=["trans_date", "small_group"],
        cols_log_norm=["amount_rur"],
        cols_identity=[],
        print_dataset_info=False,
    )

    dataset = preprocessor.fit_transform(source_data)
    assert len(dataset) == 100
    assert len(dataset[0].keys()) == 5
