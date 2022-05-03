import numpy as np

from ptls.data_load.augmentations.dropout_trx import DropoutTrx


def test_no_dropout():
    i_filter = DropoutTrx(0.0)
    data = {'mcc': np.arange(100)}
    data = i_filter(data)
    assert len(data['mcc']) == 100
    assert (np.diff(data['mcc']) >= 0).all()


def test_with_dropout():
    i_filter = DropoutTrx(0.1)
    data = {'mcc': np.arange(100)}
    data = i_filter(data)
    assert len(data['mcc']) == 91
    assert (np.diff(data['mcc']) >= 0).all()
