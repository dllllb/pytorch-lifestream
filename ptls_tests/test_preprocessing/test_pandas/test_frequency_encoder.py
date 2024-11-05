import pandas as pd
import pytest

from ptls.preprocessing.pandas.pandas_transformation.pandas_freq_transformer import FrequencyEncoder


@pytest.fixture()
def get_df_and_encoder():
    df = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1, 1],
        'cat': [4, 5, 5, 5, 2, 2, 2, 2],
    })
    t = FrequencyEncoder(col_name_original='cat')
    return df, t


def test_fit(get_df_and_encoder):
    df, t = get_df_and_encoder
    t.fit(df)
    assert t.mapping == {'2': 1, '5': 2, '4': 3}
    assert t.other_values_code == 4
    assert t.dictionary_size == 5


def test_fit_transform(get_df_and_encoder):
    df, t = get_df_and_encoder
    out = t.fit_transform(df)
    assert out['cat'].values.tolist() == [3, 2, 2, 2, 1, 1, 1, 1]


def test_new(get_df_and_encoder):
    df_train, t = get_df_and_encoder

    df_test = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1, 1],
        'cat': [4, 5, 5, 5, 3, 3, 3, 3],
    })
    t = FrequencyEncoder(col_name_original='cat')
    t.fit(df_train)
    out = t.transform(df_test)
    assert out['cat'].values.tolist() == [3, 2, 2, 2, 4, 4, 4, 4]
