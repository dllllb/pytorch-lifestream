import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


def to_event_sequence(arr, uid):
    uids = uid.reshape(-1, 1).repeat(arr.shape[1], 1)
    poss = np.arange(0, arr.shape[1], dtype=np.int16).reshape(1, -1).repeat(arr.shape[0], 0)
    ixs = arr > 0
    amnts = np.ones(ixs.sum(), dtype=np.int16)

    df = pa.Table.from_arrays([
        arr[ixs],
        uids[ixs],
        poss[ixs],
        amnts,
    ], names=['amino_acid', 'uid', 'pos', 'amnt'])
    return df


if __name__ == '__main__':
    X_train = np.load('data/X_train.npy')
    len_train = X_train.shape[0]
    uid_train = np.arange(0, len_train)
    table_train = to_event_sequence(X_train, uid_train)
    pq.write_table(table_train, 'data/X_train.parquet')
    del table_train
    del X_train

    X_test = np.load('data/X_test.npy')
    len_test = X_test.shape[0]
    uid_test = np.arange(len_train, len_train + len_test)
    table_test = to_event_sequence(X_test, uid_test)
    pq.write_table(table_test, 'data/X_test.parquet')
    del table_test
    del X_test

    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')

    df_target = pd.concat([
        pd.DataFrame({'uid': uid_train, 'y': y_train}),
        pd.DataFrame({'uid': uid_test, 'y': y_test}),
    ], axis=0)
    df_target.to_csv('data/target.csv')

    df_test_ids = pd.DataFrame({'uid': uid_test})
    df_test_ids.to_csv('data/test_ids.csv')
