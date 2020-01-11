import argparse
import logging
import os
import pickle

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
    parser.add_argument('--cols_event_time', nargs='+')
    parser.add_argument('--cols_category', nargs='*', default=[])
    parser.add_argument('--cols_log_norm', nargs='*', default=[])
    parser.add_argument('--col_target', required=False, type=str)

    parser.add_argument('--output_path', type=os.path.abspath)
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
    return col.map({k: i + 1 for i, k in enumerate(col.value_counts().index)})


def trx_to_features(df_data, print_dataset_info,
                    col_client_id, cols_event_time, cols_category, cols_log_norm):
    def copy_time(rec):
        rec['event_time'] = rec['feature_arrays']['event_time']
        del rec['feature_arrays']['event_time']
        return rec

    if print_dataset_info:
        logger.info(f'Found {df_data[col_client_id].nunique()} unique clients')

    # event_time mapping
    df_event_time = df_data[cols_event_time].drop_duplicates()
    df_event_time = df_event_time.sort_values(cols_event_time)
    df_event_time['event_time'] = np.arange(len(df_event_time))
    df_data = pd.merge(df_data, df_event_time, on=cols_event_time)

    for col in cols_category:
        df_data[col] = encode_col(df_data[col])
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

    logger.info('Feature collection in progress ...')
    features = df_data \
        .assign(et_index=lambda x: x['event_time']) \
        .set_index([col_client_id, 'et_index']).sort_index() \
        .groupby(col_client_id).apply(lambda x: {k: np.array(v) for k, v in x.to_dict(orient='list').items()}) \
        .rename('feature_arrays').reset_index().to_dict(orient='records')

    features = [copy_time(r) for r in features]

    if print_dataset_info:
        feature_names = list(features[0]['feature_arrays'].keys())
        logger.info(f'Feature names: {feature_names}')

    logger.info(f'Prepared features for {len(features)} clients')
    return features


def update_with_target(features, data_path, target_files, col_client_id, col_target):
    df_target = pd.concat([pd.read_csv(os.path.join(data_path, file)) for file in target_files])
    df_target = df_target.set_index(col_client_id)
    d_clients = df_target.to_dict(orient='index')
    logger.info(f'Target loaded for {len(d_clients)} clients')

    features = [
        dict([('target', d_clients.get(rec[col_client_id], {}).get(col_target))] + list(rec.items()))
        for rec in features
    ]
    logger.info(f'Target updated for {len(features)} clients')
    return features


def save_features(df_data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(df_data, f)
    logger.info(f'Saved to: "{save_path}"')


if __name__ == '__main__':
    config = parse_args()

    if config.log_file is not None:
        handlers = [logging.StreamHandler(), logging.FileHandler(config.log_file, mode='w')]
    else:
        handlers = None
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s',
                        handlers=handlers)

    source_data = load_source_data(
        data_path=config.data_path,
        trx_files=config.trx_files,
    )

    client_features = trx_to_features(
        df_data=source_data,
        print_dataset_info=config.print_dataset_info,
        col_client_id=config.col_client_id,
        cols_event_time=config.cols_event_time,
        cols_category=config.cols_category,
        cols_log_norm=config.cols_log_norm,
    )

    if len(config.target_files) > 0 and config.col_target is not None:
        client_features = update_with_target(
            features=client_features,
            data_path=config.data_path,
            target_files=config.target_files,
            col_client_id=config.col_client_id,
            col_target=config.col_target,
        )

    save_features(
        df_data=client_features,
        save_path=config.output_path,
    )
