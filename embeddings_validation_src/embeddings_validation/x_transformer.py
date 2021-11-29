import datetime
import logging
import re

import pandas as pd

from embeddings_validation import cls_loader
from embeddings_validation.file_reader import FeatureFile
from embeddings_validation.preprocessing.category_encoder import CategoryEncoder


logger = logging.getLogger('luigi-interface')


class XTransformer:
    def __init__(self, conf, feature_name, preprocessing):
        self.conf = conf
        self.feature_name = feature_name

        self.preprocessing = None
        self.load_preprocessors(preprocessing)

        self.feature_list = None
        self.load_time = None
        self.load_features()

        self.local_info = []

    def get_df_features(self, df_target):
        features = [df.df.set_index(df.cols_id) for df in self.feature_list]
        index = df_target.df.set_index(df_target.cols_id).index
        for i, df in enumerate(features):
            not_found = len(index.difference(df.index))
            if not_found > 0:
                self.local_info.append(f'Missing {not_found} records from {len(index)} in "{i}"th feature file')

        features = [df.reindex(index=index) for df in features]
        features = [df.rename(columns={col: f'f_{i}__{col}' for col in df.columns}) for i, df in enumerate(features)]
        features = pd.concat(features, axis=1)

        _strange_col_names = [col for col in features.columns if re.match(r'^[\w\d_]+$', col) is None]
        if len(_strange_col_names) > 0:
            logger.warning(f'Smells columns names: {_strange_col_names}')

        return features

    def fit_transform(self, df_target):
        features = self.get_df_features(df_target)
        X = features
        for p in self.preprocessing:
            X = p.fit_transform(X)
        return X

    def transform(self, df_target):
        features = self.get_df_features(df_target)
        X = features
        for p in self.preprocessing:
            X = p.transform(X)
        return X

    def load_preprocessors(self, preprocessing):
        self.preprocessing = []
        for t_name, params in preprocessing:
            p = cls_loader.create(t_name, params)
            self.preprocessing.append(p)

    def load_features(self):
        _start = datetime.datetime.now()
        current_features = self.conf.features[self.feature_name]
        read_params = current_features['read_params']
        if type(read_params) is not list:
            read_params = [read_params]

        self.feature_list = [FeatureFile.read_table(self.conf, **f) for f in read_params]
        self.load_time = datetime.datetime.now() - _start

    def get_feature_fit_info(self):
        info = []
        if len(self.local_info) > 0:
            info.extend(self.local_info)
        for p in self.preprocessing:
            if type(p) is CategoryEncoder:
                info.append(f'Encoded cols: {p.cols_for_encoding_info}')
                info.append(f'Dropped cols: {p.cols_for_drop}')
        return ' '.join(info)
