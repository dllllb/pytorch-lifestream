from ptls.data_load.iterable_processing.feature_rename import FeatureRename


def test_feature_rename():
    x = {
        'mcc': [1, 2 ,3],
        'score': 1,
    }
    i_filter = FeatureRename({'score': 'target'})
    data = next(i_filter([x]))
    assert 'mcc' in data
    assert 'target' in data
    assert 'score' not in data
