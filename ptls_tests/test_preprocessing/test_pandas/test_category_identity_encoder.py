import pandas as pd
import pytest

from ptls.preprocessing.pandas.pandas_transformation.category_identity_encoder import CategoryIdentityEncoder


def test_fit():
    df = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1],
        'cat': [1, 2, 3, 4, 1, 2, 3],
    })
    t = CategoryIdentityEncoder(col_name_original='cat')
    t.fit(df)
    assert t.min_fit_index == 1
    assert t.max_fit_index == 4


def test_fit_transform():
    df = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1],
        'cat': [1, 2, 3, 4, 1, 2, 3],
    })
    t = CategoryIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_new',
        is_drop_original_col=False,
    )
    out = t.fit_transform(df)
    assert (out['cat'] == out['cat_new']).all()


def test_fit_negative_error():
    df = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1],
        'cat': [1, 2, -3, 4, 1, 2, 3],
    })
    t = CategoryIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_new',
        is_drop_original_col=False,
    )
    with pytest.raises(AttributeError):
        out = t.fit_transform(df)


def test_fit_zero_warn():
    df = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1],
        'cat': [1, 2, 0, 4, 1, 2, 3],
    })
    t = CategoryIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_new',
        is_drop_original_col=False,
    )
    with pytest.warns(UserWarning):
        out = t.fit_transform(df)


def test_transform_out_of_fit():
    df_train = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1],
        'cat': [1, 2, 1, 4, 1, 2, 3],
    })
    df_test = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1],
        'cat': [1, 2, 5, 4, 1, 2, 3],
    })
    t = CategoryIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_new',
        is_drop_original_col=False,
    )
    t.fit(df_train)
    with pytest.warns(UserWarning):
        out = t.transform(df_test)
