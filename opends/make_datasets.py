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

    parser.add_argument('--print_dataset_info', action='store_true', default=True)
    parser.add_argument('--col_client_id', type=str)
    parser.add_argument('--col_event_time', type=str)

    parser.add_argument('--output_path', type=os.path.abspath)

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


def trx_to_features(df_data, print_dataset_info, col_client_id, col_event_time):
    def copy_time(rec):
        rec['event_time'] = rec['feature_arrays']['trans_date'].copy()
        return rec

    if print_dataset_info:
        df = df_data.groupby('client_id')['trans_date'].count().value_counts().sort_index()
        with pd.option_context('display.max_rows', 9):
            logger.info(f'Trx count per clients:\nlen(trx_list) | client_count\n{df}')

    # TODO: apply here fit_transform encodings, normalizers

    logger.info('Feature collection in progress ...')
    features = df_data \
        .assign(event_time=lambda x: pd.to_datetime(x[col_event_time])) \
        .set_index([col_client_id, 'event_time']).sort_index() \
        .groupby(col_client_id).apply(lambda x: {k: np.array(v) for k, v in x.to_dict(orient='list').items()}) \
        .rename('feature_arrays').reset_index().to_dict(orient='records')

    features = [copy_time(r) for r in features]

    if print_dataset_info:
        feature_names = list(features[0]['feature_arrays'].keys())
        logger.info(f'Feature names: {feature_names}')

    logger.info(f'Prepared features for {len(features)} clients')
    return features


def save_features(df_data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(df_data, f)
    logger.info(f'Saved to: "{save_path}"')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    config = parse_args()

    source_data = load_source_data(
        data_path=config.data_path,
        trx_files=config.trx_files,
    )

    client_features = trx_to_features(
        df_data=source_data,
        print_dataset_info=config.print_dataset_info,
        col_client_id=config.col_client_id,
        col_event_time=config.col_event_time,
    )

    save_features(
        df_data=client_features,
        save_path=config.output_path,
    )
