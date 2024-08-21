from ptls.data_load.iterable_processing.numeric_fillna import NumericFillNa
import numpy as np

def get_data(id_type=int, array_type=np.array):
    return [
        {
            'uid': id_type(1),
            'seq_len': 10,
            'mcc_code': array_type([1, 2, 3, 4]),
            'amount': array_type([1, np.nan, 3, 4]),
            'numeric': array_type([1, 2, 3, 4])
        },
        {
            'uid': id_type(2),
            'seq_len': 11,
            'mcc_code': array_type([1, 2, 3, 4, 5, 6, 7]),
            'amount': array_type([1, np.nan, 3, 4, np.nan, 6, 7]),
            'numeric': array_type([1, 2, np.nan, 4, 5, 6, 7])
        },
        {
            'uid': id_type(3),
            'seq_len': 12,
            'mcc_code': array_type([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'amount': array_type([1, 2, 3, 4, np.nan, 6, 7, 8, 9]),
            'numeric': array_type([1, np.nan, 3, 4, 5, 6, 7, 8, 9])
        },
    ]

def test_numeric_fillna():
    i_filter = NumericFillNa(['amount', 'numeric'], fill_by=1)
    data = i_filter(get_data())
    data = [np.isnan(np.hstack([rec['amount'], rec['numeric']])).any() for rec in data]
    assert not any(data)