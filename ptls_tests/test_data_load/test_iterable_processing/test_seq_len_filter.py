from ptls.data_load.iterable_processing.seq_len_filter import SeqLenFilter


def get_data(id_type=int, array_type=list):
    return [
        {
            'uid': id_type(1),
            'seq_len': 10,
            'mcc_code': array_type([1, 2, 3, 4]),
            'amount': array_type([1, 2, 3, 4]),
        },
        {
            'uid': id_type(2),
            'seq_len': 11,
            'mcc_code': array_type([1, 2, 3, 4, 5, 6, 7]),
            'amount': array_type([1, 2, 3, 4, 5, 6, 7]),
        },
        {
            'uid': id_type(3),
            'seq_len': 12,
            'mcc_code': array_type([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'amount': array_type([1, 2, 3, 4, 6, 7, 8, 9]),
        },
    ]


def get_data_with_target():
    return [
        ({
            'uid': "1",
            'mcc_code': [1, 2, 3, 4],
            'amount': [1, 2, 3, 4],
        }, 1),
        ({
             'uid': "2",
             'mcc_code': [1, 2],
             'amount': [1, 2],
         }, 0),
    ]


def test_no_action():
    i_filter = SeqLenFilter(sequence_col='mcc_code')
    data = i_filter(get_data())
    data = [rec['uid'] for rec in data]
    assert data == [1, 2, 3]


def test_min_len():
    i_filter = SeqLenFilter(sequence_col='mcc_code', min_seq_len=5)
    data = i_filter(get_data())
    data = [rec['uid'] for rec in data]
    assert data == [2, 3]


def test_max_len():
    i_filter = SeqLenFilter(sequence_col='mcc_code', max_seq_len=8)
    data = i_filter(get_data())
    data = [rec['uid'] for rec in data]
    assert data == [1, 2]


def test_seq_len_field():
    i_filter = SeqLenFilter(seq_len_col='seq_len', min_seq_len=12)
    data = i_filter(get_data())
    data = [rec['uid'] for rec in data]
    assert data == [3]


def test_target_col_detection():
    i_filter = SeqLenFilter(min_seq_len=5, max_seq_len=8)
    data = i_filter(get_data())
    data = [rec['uid'] for rec in data]
    assert data == [2]


def test_target_col_detection_str():
    i_filter = SeqLenFilter(min_seq_len=5, max_seq_len=8)
    data = i_filter(get_data(id_type=str))
    data = [rec['uid'] for rec in data]
    assert data == ['2']


def test_with_target():
    i_filter = SeqLenFilter(min_seq_len=3)
    data = i_filter(get_data_with_target())
    data = [rec[0]['uid'] for rec in data]
    assert data == ['1']
