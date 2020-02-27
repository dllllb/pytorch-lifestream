import numpy as np
from pyhocon import ConfigFactory

from dltranz.agg_features import get_feature_names, make_trx_features


def get_data():
    return [
        {
            'client_id': 1,
            'target': 1,
            'event_time': np.arange(3),
            'feature_arrays': {
                'cat1': np.array([0, 2, 2]),
                'cat2': np.array([10, 20, 0]),
                'num1': np.array([100.0, 200.0, 50.0]),
                'num2': np.array([10.0, 5.0, 3.0]),
            }
        },
        {
            'client_id': 2,
            'target': 0,
            'event_time': np.arange(5),
            'feature_arrays': {
                'cat1': np.array([0, 2, 2, 1, 1]),
                'cat2': np.array([10, 20, 0, 2, 2]),
                'num1': np.array([100.0, 200.0, 50.0, 80.0, 80.0]),
                'num2': np.array([10.0, 5.0, 3.0, 4.0, 4.0]),
            }
        },
        {
            'client_id': 3,
            'target': 0,
            'event_time': np.array([]),
            'feature_arrays': {
                'cat1': np.array([], dtype=np.int),
                'cat2': np.array([], dtype=np.int),
                'num1': np.array([]),
                'num2': np.array([]),
            }
        },
    ]


def get_conf():
    params = {
        'params': {
            'trx_encoder': {
                'numeric_values': {
                    'num1': {},
                    'num2': {},
                },
                'embeddings': {
                    'cat1': {'in': 10},
                    'cat2': {'in': 40}
                },
            }
        }
    }
    return ConfigFactory.from_dict(params)


def test_get_feature_names():
    col_names = get_feature_names(get_conf())
    expected_feature_num = 11 * 2 + 2 + 10 * 3 * 2 + 40 * 3 * 2
    assert len(col_names) == expected_feature_num


def test_make_trx_features():
    seq = make_trx_features(get_data(), get_conf())
    expected_feature_num = 11 * 2 + 2 + 10 * 3 * 2 + 40 * 3 * 2 + 2

    rec = next(seq)
    assert type(rec) is dict
    assert len(rec) == expected_feature_num
    assert rec['num1_sum'] == 350.0
    assert rec['num2_sum'] == 18.0
    np.testing.assert_almost_equal(rec['cat1_X_num1_0_sum'], 100.0 / 350.0)
    np.testing.assert_almost_equal(rec['cat1_X_num1_1_sum'], 0.0 / 350.0)
    np.testing.assert_almost_equal(rec['cat1_X_num1_2_sum'], 250.0 / 350.0)
    np.testing.assert_almost_equal(rec['cat1_X_num1_9_sum'], 0.0 / 350.0)

    rec = next(seq)
    assert type(rec) is dict
    assert len(rec) == expected_feature_num
    assert rec['num1_sum'] == 510.0
    assert rec['num2_sum'] == 26.0
    np.testing.assert_almost_equal(rec['cat1_X_num1_0_sum'], 100.0 / 510.0)
    np.testing.assert_almost_equal(rec['cat1_X_num1_1_sum'], 160.0 / 510.0)
    np.testing.assert_almost_equal(rec['cat1_X_num1_2_sum'], 250.0 / 510.0)
    np.testing.assert_almost_equal(rec['cat1_X_num1_9_sum'], 0.0 / 510.0)
    np.testing.assert_almost_equal(rec['cat2_X_num1_0_sum'], 50.0 / 510.0)
    np.testing.assert_almost_equal(rec['cat1_X_num2_2_sum'], 8.0 / 26.0)
    np.testing.assert_almost_equal(rec['cat2_X_num2_20_sum'], 5.0 / 26.0)

    rec = next(seq)
    assert type(rec) is dict
    assert len(rec) == expected_feature_num
    assert rec['num1_sum'] == 0.0
    assert rec['num2_sum'] == 0.0
    np.testing.assert_almost_equal(rec['cat1_X_num1_0_sum'], 0.0)
    np.testing.assert_almost_equal(rec['cat1_X_num1_0_std'], 0.0)

