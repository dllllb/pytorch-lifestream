from ptls.data_load.datasets.parquet_file_scan import train_valid_split


def test_train_valid_split_none():
    files = [1, 2, 3, 4, 6, 5, 9, 8]
    out = train_valid_split(files, valid_rate=None, is_sorted=False)
    assert out is files


def test_train_valid_split_zero():
    files = [1, 2, 3, 4, 6, 5, 9, 8]
    out = train_valid_split(files, valid_rate=0, is_sorted=False)
    assert out is files


def test_train_valid_split_sorted():
    files = [1, 2, 3, 4, 6, 5, 9, 8]
    out = train_valid_split(files)
    assert out == [1, 2, 3, 4, 5, 6, 8, 9]


def test_train_valid_split_correct():
    files = [1, 2, 3, 4, 6, 5, 9, 8]
    train_files = train_valid_split(files, valid_rate=0.3, return_part='train', shuffle_seed=123)
    valid_files = train_valid_split(files, valid_rate=0.3, return_part='valid', shuffle_seed=123)
    for i in train_files:
        assert i not in valid_files
    for i in valid_files:
        assert i not in train_files

    assert len(train_files) + len(valid_files) == len(files)
    assert len(set(train_files).intersection(set(valid_files))) == 0


def test_train_valid_split_incorrect_non_sorted():
    files = [1, 2, 3, 4, 6, 5, 9, 8]
    train_files = train_valid_split(files, valid_rate=0.3, return_part='train', shuffle_seed=123)
    valid_files = train_valid_split(files, valid_rate=0.3, return_part='valid', shuffle_seed=124)
    assert len(set(train_files).intersection(set(valid_files))) > 0