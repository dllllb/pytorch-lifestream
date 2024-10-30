import pandas as pd
import torch

from ptls.preprocessing.base.transformation.user_group_transformer import UserGroupTransformer
import pytest


@pytest.fixture()
def data():
    return pd.DataFrame({
        'user_id': [0, 0, 0, 1, 1, 1, 1, 2, 2],
        'event_time': [0, 1, 2, 0, 1, 2, 3, 0, 1],
        'mcc': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'amount': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
    })


def test_group(data):
    df = data
    t = UserGroupTransformer(col_name_original='user_id')
    records = t.fit_transform(df)
    records = records.to_dict(orient='records')
    rec = records[1]
    assert rec['user_id'] == 1
    torch.testing.assert_close(rec['event_time'], torch.LongTensor([0, 1, 2, 3]))
    torch.testing.assert_close(rec['mcc'], torch.LongTensor([3, 4, 5, 6]))
    torch.testing.assert_close(rec['amount'], torch.DoubleTensor([13, 14, 15, 16]))


def test_group_with_target():
    df = pd.DataFrame({
        'user_id': [0, 0, 0, 1, 1, 1, 1, 2, 2],
        'event_time': [0, 1, 2, 0, 1, -1, 3, 0, 1],
        'mcc': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'target': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
    })
    t = UserGroupTransformer(col_name_original='user_id', cols_first_item=['target'])
    records = t.fit_transform(df).to_dict(orient='records')
    rec = records[1]
    assert rec['user_id'] == 1
    torch.testing.assert_close(rec['event_time'], torch.LongTensor([-1, 0, 1, 3]))
    torch.testing.assert_close(rec['mcc'], torch.LongTensor([5, 3, 4, 6]))
    assert rec['target'] == 15
