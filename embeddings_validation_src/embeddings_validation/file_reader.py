import os
import pickle

import numpy as np
import pandas as pd

ID_TYPE_MAPPING = {
    'str': str,
    'int': np.int32,
    'date': 'datetime64[D]',
    'datetime': 'datetime64[s]',
}


class BaseReader:
    def __init__(self, conf):
        self.conf = conf
        self.source_path = []
        self.df = None

        target_conf = conf['target']
        self.cols_id = [x for x in target_conf['cols_id']]
        self.cols_id_type = [ID_TYPE_MAPPING.get(x) for x in target_conf['cols_id_type']]
        self.col_target = target_conf['col_target']

        _unknown = [s for t, s in zip(self.cols_id_type, target_conf['cols_id_type']) if t is None]
        if len(_unknown) > 0:
            raise AttributeError(f'Unknown types: {_unknown}')

    def clone_schema(self):
        new = self.__class__(self.conf)
        new.source_path = self.source_path
        return new

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            file = pickle.load(f)
        return file

    @staticmethod
    def _read_pd(path, read_args):
        ext = os.path.splitext(path)[1]
        if ext == '.csv':
            return pd.read_csv(path, **read_args)
        if ext == '.pickle':
            return pd.read_pickle(path, **read_args)
        if ext == '.parquet':
            return pd.read_parquet(path, **read_args)
        raise NotImplementedError(f'Not implemented for "{ext}" file type')

    @classmethod
    def read_table(cls, conf, file_name, rename_cols=None, drop_cols=None, read_args=None,
                   drop_duplicated_ids=False, **kwargs):
        if read_args is None:
            read_args = {}

        self = cls(conf)
        columns = self.keep_columns()

        if type(file_name) is not list:
            file_name = [file_name]
        self.source_path = []
        df = []
        for f in file_name:
            path = os.path.join(conf.root_path, f)
            self.source_path.append(path)
            df.append(self._read_pd(path, read_args))
        self.df = pd.concat(df, axis=0)

        if rename_cols is not None:
            self.df = self.df.rename(columns=rename_cols)
        if drop_cols is not None:
            self.df = self.df.drop(columns=drop_cols)

        if columns is not None:
            self.df = self.df[columns]
        self.cols_id_type_cast()
        if drop_duplicated_ids:
            self.drop_duplicated_ids()
        else:
            self.check_duplicated_ids()

        return self

    def __len__(self):
        return len(self.df)

    def keep_columns(self):
        raise NotImplementedError()

    def cols_id_type_cast(self):
        for col, dtype in zip(self.cols_id, self.cols_id_type):
            self.df[col] = self.df[col].astype(dtype)

    def drop_duplicated_ids(self):
        self.df = self.df.drop_duplicates(subset=self.cols_id, keep='first', ignore_index=True)

    def check_duplicated_ids(self):
        duplicated_count = self.df.duplicated(subset=self.cols_id, keep=False).sum()
        if duplicated_count > 0:
            raise IndexError(f'Found {duplicated_count} duplicated rows in "{self.source_path}". '
                             f'Check and fix your dataset or use `drop_duplicated_ids=True`')

    @property
    def target_values(self):
        return self.df[self.col_target]

    @property
    def ids_values(self):
        return self.df[self.cols_id]

    def select_ids(self, df_ids):
        new_df = self.clone_schema()
        new_df.df = pd.merge(self.df, df_ids.ids_values, left_on=self.cols_id, right_on=df_ids.cols_id, how='inner')
        return new_df

    def exclude_ids(self, df_ids):
        new_df = self.clone_schema()
        df = pd.merge(self.df, df_ids.ids_values,
                      left_on=self.cols_id, right_on=df_ids.cols_id, how='left', indicator=True)
        excluded_cnt = df['_merge'].eq('both').sum()
        df = df[df['_merge'].eq('left_only')].drop(columns='_merge')
        new_df.df = df
        return new_df, int(excluded_cnt)

    def select_pos(self, df_pos):
        new_df = self.clone_schema()
        new_df.df = self.df.iloc[df_pos]
        return new_df


class TargetFile(BaseReader):
    def keep_columns(self):
        return self.cols_id + [self.col_target]


class IdFile(BaseReader):
    def keep_columns(self):
        return self.cols_id


class FeatureFile(BaseReader):
    def keep_columns(self):
        return None
