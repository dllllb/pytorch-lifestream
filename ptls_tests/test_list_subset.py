from ptls.util import ListSubset


def test_list_subset_iter():
    ls = ListSubset([1, 2, 3], [0, 2])
    res = [e for e in ls]
    assert [1, 3] == res


def test_list_subset_geitem():
    ls = ListSubset([1, 2, 3], [0, 2])
    assert len(ls) == 2
    assert ls[1] == 3
