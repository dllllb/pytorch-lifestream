from pathlib import Path

import pandas as pd
import torch

from ptls.data_load.datasets import DuckDbDataset


def test_simple_processing():
    test_df = pd.DataFrame([
        {
            'id': 1,
            'dt': 1223,
            'sum': 20,
            'category': 3
        },
        {
            'id': 2,
            'dt': 1212,
            'sum': 43,
            'category': 4
        },
        {
            'id': 1,
            'dt': 1010,
            'sum': 100,
            'category': 12
        },
        {
            'id': 1,
            'dt': 1011,
            'sum': 50,
            'category': 3
        },
    ])

    source = f"""
        (SELECT * FROM read_csv_auto('{Path(__file__).parent / 'test_df.csv'}'))
        """

    ds = DuckDbDataset(
        data_read_func=source,
        col_id='id',
        col_event_time='dt',
        col_event_fields=['sum', 'category']
    )

    expected = str([
        {
            'id': 1,
            'sum': torch.tensor([100, 50, 20]),
            'category': torch.tensor([12, 3, 3]),
            'dt': torch.tensor([1010, 1011, 1223])
        },
        {
            'id': 2,
            'sum': torch.tensor([43]),
            'category': torch.tensor([4]),
            'dt': torch.tensor([1212])
        }
    ])

    recs = [rec for rec in ds]

    assert str(recs) == expected
