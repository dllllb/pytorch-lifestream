import hashlib
import os
import logging
from functools import reduce
from operator import iadd

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


RANDOM_SEED = 42
DATA_PATH = 'test_data'


def to_csv(df, file_name):
    path = os.path.join(DATA_PATH, file_name)
    df.to_csv(path, index=False)
    print(f'Saved {file_name} to "{path}". Shape: {df.shape}')


def create_separate_files(n_train=10000, n_valid=1000, n_test=1000, num_features_list=(10, 20)):
    """
    Create:
        - 2 feature files with 10 and 20 features and id-column
        - target tile with id-col and target-col
        - 3 split files (train valid test) with id-cols

    :return:
    """
    print('create_separate_files start')
    rs = np.random.RandomState(RANDOM_SEED)
    all_ids = np.arange(0, n_train + n_valid + n_test) + 10000
    rs.shuffle(all_ids)

    id_train = all_ids[:n_train]
    id_valid = all_ids[n_train:n_train + n_valid]
    id_test = all_ids[-n_test:]

    df_train = pd.DataFrame({'id': id_train}).sort_values('id')
    df_valid = pd.DataFrame({'id': id_valid}).sort_values('id')
    df_test = pd.DataFrame({'id': id_test}).sort_values('id')

    to_csv(df_train, 'id_train.csv')
    to_csv(df_valid, 'id_valid.csv')
    to_csv(df_test, 'id_test.csv')

    df_target = []
    for i, n_features in enumerate(num_features_list):
        columns = [f'col_{j}' for j in range(n_features)]
        df_features = pd.DataFrame(rs.randn(all_ids.shape[0], n_features).round(3), columns=columns)
        col_id = 'id' if i != 0 else 'uid'
        df_features[col_id] = all_ids
        to_csv(df_features, f'features_{i}.csv')

        for _ in range(i + 1):
            col_target = rs.choice(columns, 1)[0]
            df_target.append(df_features[col_target].clip(-1, None))
    df_target.append(pd.Series(rs.randn(all_ids.shape[0])))
    df_target = reduce(iadd, df_target) / len(df_target)
    df_target = (df_target > 0.5).astype(int)
    df_target = pd.DataFrame({'id': all_ids, 'y': df_target}).sort_values('id')
    to_csv(df_target, 'target.csv')
    print('Mean target: {:.3f}'.format(df_target['y'].mean()))


def create_single_file(
        n_rows=20000, extra_ids=0.20, missing_ids=0.10,
        missing_values_rate=0.01, num_features=10, cat_features=(2, 5, 8),
        hash_fextures=2,
        num_embeddings=20,
):
    """
    Create:
        - baseline file with id column, target columns and baseline columns of different types
        - 1 feature files with 20 features and id column
        - feature file has shifted ids, some ids are missing, some ids are new

    :return:
    """
    print('create_single_file start')

    rs = np.random.RandomState(RANDOM_SEED)
    uids = np.arange(int(n_rows * (1 + extra_ids))) + 10
    columns = {'uid': rs.choice(uids, n_rows, replace=False)}

    # random error
    target_column = [rs.randn(n_rows) * 5]

    for i in range(num_features):
        target_values = rs.randn(n_rows).round(2)
        if i == 0:
            target_column.append(target_values)
        target_values[rs.choice(int(n_rows * missing_values_rate))] = np.NaN
        columns[f'num_{i:02d}'] = target_values

    for i, n in enumerate(cat_features):
        target_values = rs.choice(n, n_rows)
        if i <= 1:
            target_column.append(target_values)

        values_dict = np.array([f'v{j}' for j in range(n)]).astype(object)
        target_values = values_dict[target_values]
        target_values[rs.choice(int(n_rows * missing_values_rate))] = np.NaN
        columns[f'cat_{i:02d}'] = target_values

    for i in range(hash_fextures):
        target_values = columns['uid']
        target_values = target_values.astype(str).astype(object)
        target_values += 'ABCDEF' + str(i)
        target_values = [hashlib.md5(v.encode()).hexdigest() for v in target_values]
        columns[f'hash_{i:02d}'] = target_values

    df_baseline = pd.DataFrame(columns)

    columns = [f'col_{j}' for j in range(num_embeddings)]
    df_embeddings = pd.DataFrame(rs.randn(uids.shape[0], num_embeddings).round(4), columns=columns)
    df_embeddings['uid'] = uids
    for i in range(2):
        target_column.append(df_embeddings.set_index('uid').reindex(index=df_baseline['uid']).iloc[:, i].values)

    df_embeddings = df_embeddings.sample(n=int(n_rows * (1 - missing_ids)), random_state=RANDOM_SEED)
    to_csv(df_embeddings, 'embeddings.csv')

    target_values = np.concatenate([c.reshape(-1, 1) for c in target_column], axis=1)
    target_values *= rs.rand(1, target_values.shape[1])
    target_values = target_values.sum(axis=1)
    target_values = (target_values > 0.0).astype(int)
    df_baseline['target'] = target_values
    to_csv(df_baseline, 'baseline.csv')
    print('Mean target: {:.3f}'.format(df_baseline['target'].mean()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s')

    create_separate_files()
    create_single_file()

    print('done')
