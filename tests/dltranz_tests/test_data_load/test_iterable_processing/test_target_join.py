from ptls.data_load.iterable_processing.target_join import TargetJoin


def get_features_only():
    return [
        {'id': 1},
        {'id': 2},
        {'id': 3},
        {'id': 4},
    ]


def get_features_with_incorrect_target():
    return [
        ({'id': 1}, 0),
        ({'id': 2}, 1),
        ({'id': 3}, 0),
        ({'id': 4}, 1),
    ]

def get_targets():
    return {1: 0, 2: 0, 3: 1, 4: 1}


def test_join():
    i_filter = TargetJoin('id', get_targets())
    data = i_filter(get_features_only())
    data = [y for x, y in data]
    assert data == [0, 0, 1, 1]


def test_join_and_replace():
    i_filter = TargetJoin('id', get_targets())
    orig_data = get_features_with_incorrect_target()

    data = orig_data
    data = [y for x, y in data]
    assert data == [0, 1, 0, 1]  # without replace

    data = i_filter(orig_data)
    data = [y for x, y in data]
    assert data == [0, 0, 1, 1]  # with replace


