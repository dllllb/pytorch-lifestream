from ptls.data_load.iterable_processing.target_extractor import TargetExtractor


def get_features_only():
    return [
        {'id': 1, 'target': 0},
        {'id': 2, 'target': 0},
        {'id': 3, 'target': 1},
        {'id': 4, 'target': 1},
    ]


def get_features_with_incorrect_target():
    return [
        ({'id': 1, 'target': 0}, 0),
        ({'id': 2, 'target': 0}, 1),
        ({'id': 3, 'target': 1}, 0),
        ({'id': 4, 'target': 1}, 1),
    ]


def test_extract():
    i_filter = TargetExtractor('target')
    data = i_filter(get_features_only())
    data = [y for x, y in data]
    assert data == [0, 0, 1, 1]


def test_extract_and_replace():
    i_filter = TargetExtractor('target')
    orig_data = get_features_with_incorrect_target()

    data = orig_data
    data = [y for x, y in data]
    assert data == [0, 1, 0, 1]  # without replace

    data = i_filter(orig_data)
    data = [y for x, y in data]
    assert data == [0, 0, 1, 1]  # with replace


