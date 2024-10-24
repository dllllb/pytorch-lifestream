import pandas as pd

from ptls.preprocessing.base.transformation.col_identity_transformer import ColIdentityEncoder


def test_identity():
    df = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1],
        'cat': [1, 2, 0, 4, 1, 2, 3],
    })
    t = ColIdentityEncoder(
        col_name_original='cat',
    )
    out = t.fit_transform(df)
    assert (df == out).all().all()


def test_new_col():
    df = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1],
        'cat': [1, 2, 0, 4, 1, 2, 3],
    })
    t = ColIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_2',
        is_drop_original_col=False,
    )
    out = t.fit_transform(df)
    assert (df['cat'] == out['cat_2']).all()
    assert (df['cat'] == out['cat']).all()


def test_rename():
    df = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1],
        'cat': [1, 2, 0, 4, 1, 2, 3],
    })
    t = ColIdentityEncoder(
        col_name_original='cat',
        col_name_target='cat_2',
        is_drop_original_col=True,
    )
    out = t.fit_transform(df)
    assert (df['cat'] == out['cat_2']).all()
    assert 'cat' not in out.columns
