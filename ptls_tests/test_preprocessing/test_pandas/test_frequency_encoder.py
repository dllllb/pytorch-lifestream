import pandas as pd
from ptls.preprocessing.pandas.frequency_encoder import FrequencyEncoder


def test_fit():
    df = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1, 1],
        'cat': [4, 5, 5, 5, 2, 2, 2, 2],
    })
    t = FrequencyEncoder(col_name_original='cat')
    t.fit(df)
    assert t.mapping == {'2': 1, '5': 2, '4': 3}
    assert t.other_values_code == 4
    assert t.dictionary_size == 5


def test_fit_transform():
    df = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1, 1],
        'cat': [4, 5, 5, 5, 2, 2, 2, 2],
    })
    t = FrequencyEncoder(col_name_original='cat')
    out = t.fit_transform(df)
    assert out['cat'].values.tolist() == [3, 2, 2, 2, 1, 1, 1, 1]


def test_new():
    df_train = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1, 1],
        'cat': [4, 5, 5, 5, 2, 2, 2, 2],
    })
    df_test = pd.DataFrame({
        'uid': [0, 0, 0, 1, 1, 1, 1, 1],
        'cat': [4, 5, 5, 5, 3, 3, 3, 3],
    })
    t = FrequencyEncoder(col_name_original='cat')
    t.fit(df_train)
    out = t.transform(df_test)
    assert out['cat'].values.tolist() == [3, 2, 2, 2, 4, 4, 4, 4]
