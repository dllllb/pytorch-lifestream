import logging
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)

DATA_LOG = 'small_data'
DATA_USERS = 'parquet_embeddings'
DATA_ITEMS = 'stories_texts_embeddings'


def list_src_files(data_path, load_all_data):
    file_list = sorted(os.listdir(data_path))
    logger.info(f'Found {len(file_list)} files in "{data_path}"')
    if not load_all_data:
        file_list = file_list[:1] + file_list[-1:]
        logger.info(f'Read {len(file_list)} files from "{data_path}" in debug mode')
    return [os.path.join(data_path, f) for f in file_list]


def read_parquet(file_name):
    py_script = f"""
import pandas as pd
import pickle
import sys

df = pd.read_parquet('{file_name}')
sys.stdout.buffer.write(pickle.dumps(df))
    """
    p = subprocess.Popen([sys.executable, '-c', py_script], stdout=subprocess.PIPE)
    data, _ = p.communicate()
    return pickle.loads(data)


def load_data(config):
    df_log = pd.read_csv(os.path.join(config.data_path, 'stories_reaction_train.csv'))
    df_log['event_dttm'] = pd.to_datetime(df_log['event_dttm'])
    return df_log


def log_split_by_date(df, test_size):
    split_pos = int(len(df) * test_size)
    df = df.sort_values('event_dttm')
    df_train, df_valid = df.iloc[:split_pos], df.iloc[split_pos:]
    logger.info(f'df_log ({df.shape}) split on ({df_train.shape}) and ({df_valid.shape})')
    return df_train, df_valid


def get_encoder(df_log, col):
    all_items = df_log[col].value_counts()
    logger.info(f'Found {len(all_items)} {col}')
    encoded_items = all_items.index.tolist()[:int(len(all_items) * 0.95)]
    logger.info(f'{len(encoded_items)} {col} will be encoded')
    item_encoder = {v: k + 1 for k, v in enumerate(encoded_items)}
    logger.info(f'encoder for {col} : {len(item_encoder)} keys. '
                f'Min: {min(item_encoder.values())}, max: {max(item_encoder.values())}')
    return item_encoder


class StoriesDataset(torch.utils.data.Dataset):
    def __init__(self, df_log, df_user, user_encoder, df_item, item_encoder):
        self.df_user = df_user
        self.user_encoder = user_encoder
        self.df_item = df_item
        self.item_encoder = item_encoder

        if self.df_user is not None:
            logger.info(f'Found {self.df_user.size} features in users')
        else:
            logger.info(f'User features not found')

        if self.df_item is not None:
            logger.info(f'Found {self.df_item.size} features in items')
        else:
            logger.info(f'Item features not found')

        self.df_log = df_log.assign(
            reward=df_log['event'].map({'dislike': -10.0, 'skip': -0.1, 'view': 0.1, 'like': 0.5}))

    def __len__(self):
        return len(self.df_log)

    def __getitem__(self, item):
        item_obj = self.df_log.iloc[item]

        user_id = item_obj['customer_id']
        item_id = item_obj['story_id']

        if self.df_user is not None:
            f_user = self.df_user.get_item(user_id)
        else:
            f_user = np.array([], dtype=np.float32)

        if self.df_item is not None:
            f_item = self.df_item.get_item(item_id)
        else:
            f_item = np.array([], dtype=np.float32)

        user_encoded_id = np.int32(self.user_encoder.get(user_id, 0))
        item_encoded_id = np.int16(self.item_encoder.get(item_id, 0))

        return f_user, user_encoded_id, f_item, item_encoded_id, np.float32(item_obj['reward'])


class RandomSampler(Sampler):
    def __init__(self, n_total, n_sample, target):
        super().__init__(None)

        self.n_total = int(n_total)
        self.n_sample = int(n_sample)
        self.target = target

        ix = np.arange(self.n_total)
        self.pos_ix = ix[self.target == 1]
        self.neg_ix = ix[self.target == 0]

        self.neg_cnt = min(self.n_sample // 2, len(self.pos_ix), len(self.neg_ix))
        self.pos_cnt = self.neg_cnt

    def __len__(self):
        return self.pos_cnt + self.neg_cnt

    def __iter__(self):
        np.random.shuffle(self.pos_ix)
        np.random.shuffle(self.neg_ix)

        ix = np.concatenate([self.pos_ix[:self.pos_cnt], self.neg_ix[:self.neg_cnt]])
        np.random.shuffle(ix)
        return iter(ix.tolist())
