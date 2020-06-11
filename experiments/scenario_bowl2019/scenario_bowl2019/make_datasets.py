import argparse
import datetime
import json
import logging
import os
import pickle
from copy import copy
from random import Random

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=os.path.abspath)
    parser.add_argument('--trx_files', nargs='+')
    parser.add_argument('--target_files', nargs='*', default=[])

    parser.add_argument('--print_dataset_info', action='store_true', default=True)
    parser.add_argument('--col_client_id', type=str)
    parser.add_argument('--col_sample_id', type=str)
    parser.add_argument('--cols_event_time', nargs='+')
    parser.add_argument('--cols_category', nargs='*', default=[])
    parser.add_argument('--cols_log_norm', nargs='*', default=[])
    parser.add_argument('--cols_identity', nargs='*', default=[])
    parser.add_argument('--cols_enumerate', nargs='*', default=[])
    parser.add_argument('--cols_event_data_parse', nargs='*', default=[])
    parser.add_argument('--col_target', required=False, type=str)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--salt', type=int, default=42)

    parser.add_argument('--output_train_trx_path', type=os.path.abspath)
    parser.add_argument('--output_train_mapping_path', type=os.path.abspath)
    parser.add_argument('--output_train_path', type=os.path.abspath)
    parser.add_argument('--output_test_path', type=os.path.abspath)
    parser.add_argument('--output_test_ids_path', type=os.path.abspath)
    parser.add_argument('--log_file', type=os.path.abspath)

    args = parser.parse_args(args)
    logger.info('Parsed args:\n' + '\n'.join([f'  {k:15}: {v}' for k, v in vars(args).items()]))
    return args


def load_source_data(data_path, trx_files):
    data = []
    for file in trx_files:
        file_path = os.path.join(data_path, file)
        df = pd.read_csv(file_path)
        data.append(df)
        logger.info(f'Loaded {len(df)} rows from "{file_path}"')

    data = pd.concat(data, axis=0)
    logger.info(f'Loaded {len(data)} rows in total')
    return data


def pd_hist(data, name, bins=10):
    if data.dtype.kind == 'f':
        bins = np.linspace(data.min(), data.max(), bins + 1).round(1)
    elif np.percentile(data, 99) - data.min() > bins - 1:
        bins = np.linspace(data.min(), np.percentile(data, 99), bins).astype(int).tolist() + [int(data.max() + 1)]
    else:
        bins = np.arange(data.min(), data.max() + 2, 1).astype(int)
    df = pd.cut(data, bins, right=False).rename(name)
    df = df.to_frame().assign(cnt=1).groupby(name)[['cnt']].sum()
    df['% of total'] = df['cnt'] / df['cnt'].sum()
    return df


def encode_col(col):
    col = col.astype(str)
    mapping = {k: i + 1 for i, k in enumerate(col.value_counts().index)}
    return col.map(mapping), mapping


def encode_col_with_mapping(col, mapping):
    col = col.astype(str)
    return col.map(mapping)


def trx_to_features(df_data, config, mapping_dict=None):
    def copy_time(rec):
        rec['event_time'] = rec['feature_arrays']['event_time']
        del rec['feature_arrays']['event_time']
        return rec

    def _td_default(df, cols_event_time):
        df_event_time = df[cols_event_time].drop_duplicates()
        df_event_time = df_event_time.sort_values(cols_event_time)
        df_event_time['event_time'] = np.arange(len(df_event_time))
        df = pd.merge(df, df_event_time, on=cols_event_time)
        logger.info('Default time transformation')
        return df

    def _td_datetime_string(df, col_event_time):
        df['event_time'] = pd.to_datetime(df[col_event_time]).dt.tz_localize(None)
        df['event_time'] = (df['event_time'] - np.datetime64('1970-01-01')) / np.timedelta64(1,'D')
        logger.info('To-float time transformation')
        return df

    def _td_float(df, col_event_time):
        df['event_time'] = df[col_event_time].astype(float)
        logger.info('To-float time transformation')
        return df

    def enumerate_values(s: pd.Series):
        return s.map({k:v for v,k in enumerate(s.unique())})

    print_dataset_info = config.print_dataset_info
    col_client_id = config.col_client_id
    cols_event_time = config.cols_event_time
    cols_category = config.cols_category
    cols_log_norm = config.cols_log_norm
    cols_identity = config.cols_identity
    cols_enumerate = config.cols_enumerate
    cols_event_data_parse = config.cols_event_data_parse

    if print_dataset_info:
        logger.info(f'Found {df_data[col_client_id].nunique()} unique clients')

    # event_time mapping
    if cols_event_time[0][0] == '#':
        if cols_event_time[0] == '#float':
            df_data = _td_float(df_data, cols_event_time[1])
        elif cols_event_time[0] == '#datetime_string':
            df_data = _td_datetime_string(df_data, cols_event_time[1])
        elif cols_event_time[0] == '#gender':
            df_data = _td_gender(df_data, cols_event_time[1])
        else:
            raise NotImplementedError(f'Unknown type of data transformation: "{cols_event_time[0]}"')
    else:
        df_data = _td_default(df_data, cols_event_time)

    for col in cols_event_data_parse:
        df_data[col] = df_data['event_data'].apply(lambda x: json.loads(x).get(col, 'None'))

    use_mapping = (mapping_dict is not None)
    if use_mapping:
        for col in cols_category:
            df_data[col] = encode_col_with_mapping(df_data[col], mapping_dict[col])
            if print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{pd_hist(df_data[col], col)}')
    else:
        mapping_dict = {}
        for col in cols_category:
            df_data[col], mapping = encode_col(df_data[col])
            mapping_dict[col] = mapping
            if print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{pd_hist(df_data[col], col)}')

    for col in cols_log_norm:
        df_data[col] = np.log1p(abs(df_data[col])) * np.sign(df_data[col])
        df_data[col] /= abs(df_data[col]).max()
        if print_dataset_info:
            logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{pd_hist(df_data[col], col)}')

    if print_dataset_info:
        df = df_data.groupby(col_client_id)['event_time'].count()
        logger.info(f'Trx count per clients:\nlen(trx_list) | client_count\n{pd_hist(df, "trx_count")}')

    for col in cols_enumerate:
        df_data[col] = df_data.groupby(col_client_id)[col].transform(enumerate_values)

    # column filter
    used_columns = cols_category + cols_log_norm + cols_identity + cols_enumerate + ['event_time', col_client_id]
    used_columns = [col for col in df_data.columns if col in used_columns]

    logger.info('Feature collection in progress ...')
    features = df_data[used_columns] \
        .assign(et_index=lambda x: x['event_time']) \
        .set_index([col_client_id, 'et_index']).sort_index() \
        .groupby(col_client_id).apply(lambda x: {k: np.array(v) for k, v in x.to_dict(orient='list').items()}) \
        .rename('feature_arrays').reset_index().to_dict(orient='records')

    features = [copy_time(r) for r in features]

    if print_dataset_info:
        feature_names = list(features[0]['feature_arrays'].keys())
        logger.info(f'Feature names: {feature_names}')

    logger.info(f'Prepared features for {len(features)} clients')
    if use_mapping:
        return features
    else:
        return features, mapping_dict


def update_with_target_bowl(source_data, features, data_path, target_files, col_client_id):
    df_target = pd.concat([pd.read_csv(os.path.join(data_path, file)) for file in target_files])

    data = source_data.merge(df_target[['game_session','accuracy_group']], how = 'right')
    data = data[(data.event_type == 'Assessment') & (data.event_code == 2000)] \
        [[col_client_id, 'game_session', 'timestamp', 'accuracy_group']]
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
    data['timestamp'] = (data['timestamp'] - np.datetime64('1970-01-01')) / np.timedelta64(1,'D')

    client_features = {x[col_client_id]:x for x in features}
    dataset = []
    for _, row in data.iterrows():
        train_row = copy(client_features[row[col_client_id]])
        train_row['target'] = row['accuracy_group']
        train_row['game_session'] = row['game_session']
        last_event = np.searchsorted(train_row['event_time'], row['timestamp']) + 1
        train_row['event_time'] = train_row['event_time'][:last_event]
        train_row['feature_arrays'] = {k:v[:last_event] for k,v in train_row['feature_arrays'].items()}
        dataset.append(train_row)

    return dataset


def split_dataset(all_data, test_size, data_path, target_files, col_client_id, salt):
    df_target = pd.concat([pd.read_csv(os.path.join(data_path, file)) for file in target_files])
    s_clients = set(df_target[col_client_id].tolist())

    # shuffle client list
    s_all_data_clients = set(rec[col_client_id] for rec in all_data)
    s_clients = (cl_id for cl_id in s_clients if cl_id in s_all_data_clients)
    s_clients = sorted(s_clients)
    s_clients = [cl_id for cl_id in s_clients]
    Random(salt).shuffle(s_clients)

    # split client list
    Nrows_test = int(len(s_clients) * test_size)
    s_clients_train = s_clients[:-Nrows_test]
    s_clients_test = s_clients[-Nrows_test:]

    # split data
    labeled_train = [rec for rec in all_data if rec[col_client_id] in s_clients_train]
    labeled_test = [rec for rec in all_data if rec[col_client_id] in s_clients_test]
    unlabeled = [rec for rec in all_data if rec[col_client_id] not in s_clients]
    train = labeled_train + unlabeled
    test = labeled_test

    logger.info(f'Train size: {len(train)} clients')
    logger.info(f'Test size: {len(test)} clients')

    return train, test


def save_features(df_data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(df_data, f)
    logger.info(f'Saved to: "{save_path}"')


if __name__ == '__main__':
    _start = datetime.datetime.now()
    config = parse_args()

    if config.log_file is not None:
        handlers = [logging.StreamHandler(), logging.FileHandler(config.log_file, mode='w')]
    else:
        handlers = None
    logging.basicConfig(level=logging.INFO, format='%(funcName)-20s   : %(message)s',
                        handlers=handlers)

    source_data = load_source_data(
        data_path=config.data_path,
        trx_files=config.trx_files,
    )

    client_features, mapping_dict = trx_to_features(
        df_data=source_data.copy(),
        config=config
    )
    with open(config.output_train_mapping_path, 'wb') as f:
        pickle.dump(mapping_dict, f)
    logger.info(f'Encoding col dict saved to: "{config.output_train_mapping_path}"')

    if config.test_size > 0:
        train_trx, test_trx = split_dataset(
            all_data=client_features,
            test_size=config.test_size,
            data_path=config.data_path,
            target_files=config.target_files,
            col_client_id=config.col_client_id,
            salt=config.salt,
        )
    else:
        train_trx = client_features

    save_features(
        df_data=train_trx,
        save_path=config.output_train_trx_path,
    )

    target_dataset = update_with_target_bowl(
        source_data=source_data,
        features=client_features,
        data_path=config.data_path,
        target_files=config.target_files,
        col_client_id=config.col_client_id,
    )

    if config.test_size > 0:
        test_ids = set([rec[config.col_client_id] for rec in test_trx])
        test = [x for x in target_dataset if x[config.col_client_id] in test_ids]
        train = [x for x in target_dataset if x[config.col_client_id] not in test_ids]

        save_features(
            df_data=train,
            save_path=config.output_train_path,
        )

        save_features(
            df_data=test,
            save_path=config.output_test_path,
        )
        test_ids = pd.DataFrame({config.col_sample_id: [rec[config.col_sample_id] for rec in test]})
        test_ids.to_csv(config.output_test_ids_path, index=False)

    else:
        save_features(
            df_data=target_dataset,
            save_path=config.output_train_path,
        )

    _duration = datetime.datetime.now() - _start
    logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')
