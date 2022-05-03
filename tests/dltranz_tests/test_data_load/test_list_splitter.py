import pytest

from ptls.data_load.list_splitter import ListSplitter


def test_data_size_select_01():
    data = ListSplitter([f'{i}.csv' for i in range(10)], valid_size=0.5)
    sizes = data.size_select()
    assert sizes == (5, 5)


def test_data_size_select_02():
    data = ListSplitter([f'{i}.csv' for i in range(10)], valid_size=0.05)
    sizes = data.size_select()
    assert sizes == (9, 1)


def test_data_size_select_03():
    data = ListSplitter([f'{i}.csv' for i in range(2)], valid_size=0.05)
    sizes = data.size_select()
    assert sizes == (1, 1)


def test_data_size_select_04():
    with pytest.raises(AttributeError):
        _ = ListSplitter([f'{i}.csv' for i in range(1)], valid_size=0.05)


def test_data_files():
    data = ListSplitter([f'{i}.csv' for i in range(5)], valid_size=0.45, seed=42)
    assert data.valid == ['1.csv', '4.csv']
