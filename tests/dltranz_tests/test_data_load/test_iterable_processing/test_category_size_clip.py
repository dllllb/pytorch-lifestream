from ptls.data_load.iterable_processing.category_size_clip import CategorySizeClip
import numpy as np


def get_data_without_target():
    return [
        {
            'mcc': np.array([0, 1, 2, 5, 8, 9, 11, 15]),
            'currency': np.array([0, 1, 2, 5, 8, 9, 11, 15]),
        }
    ]


def get_data_with_target():
    return [
        ({
            'mcc': np.array([0, 1, 2, 5, 8, 9, 11, 15]),
            'currency': np.array([0, 1, 2, 5, 8, 9, 11, 15]),
        }, 1)
    ]


def test_max_clip_without_target():
    i_filter = CategorySizeClip({'mcc': 10, 'currency': 4})
    data = i_filter(get_data_without_target())
    data = [x for x in data][0]
    np.testing.assert_equal(data['mcc'], np.array([0, 1, 2, 5, 8, 9, 9, 9]))
    np.testing.assert_equal(data['currency'], np.array([0, 1, 2, 3, 3, 3, 3, 3]))


def test_max_clip_with_target():
    i_filter = CategorySizeClip({'mcc': 10, 'currency': 4})
    data = i_filter(get_data_with_target())
    data = [x for x in data][0][0]
    np.testing.assert_equal(data['mcc'], np.array([0, 1, 2, 5, 8, 9, 9, 9]))
    np.testing.assert_equal(data['currency'], np.array([0, 1, 2, 3, 3, 3, 3, 3]))


def test_value_clip_without_target():
    i_filter = CategorySizeClip({'mcc': 10, 'currency': 4}, replace_value=1)
    data = i_filter(get_data_without_target())
    data = [x for x in data][0]
    np.testing.assert_equal(data['mcc'], np.array([0, 1, 2, 5, 8, 9, 1, 1]))
    np.testing.assert_equal(data['currency'], np.array([0, 1, 2, 1, 1, 1, 1, 1]))


def test_value_clip_with_target():
    i_filter = CategorySizeClip({'mcc': 10, 'currency': 4}, replace_value=1)
    data = i_filter(get_data_with_target())
    data = [x for x in data][0][0]
    np.testing.assert_equal(data['mcc'], np.array([0, 1, 2, 5, 8, 9, 1, 1]))
    np.testing.assert_equal(data['currency'], np.array([0, 1, 2, 1, 1, 1, 1, 1]))

