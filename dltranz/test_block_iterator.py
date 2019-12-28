from dltranz.train import block_iterator


def test_1():
    source = range(5)
    res = [i for i in block_iterator(source, 2)]
    assert [[0, 1], [2, 3], [4]] == res


def test_2():
    source = range(5)
    res = [i for i in block_iterator(source, 5)]
    assert [[0, 1, 2, 3, 4]] == res


def test_3():
    source = range(5)
    res = [i for i in block_iterator(source, 10)]
    assert [[0, 1, 2, 3, 4]] == res
