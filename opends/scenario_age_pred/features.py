import os

import numpy as np
import pandas as pd


def _random_features(conf):
    all_clients = pd.read_csv(os.path.join(conf['data_path'], 'train_target.csv')).set_index('client_id')
    all_clients = all_clients.assign(random=np.random.rand(len(all_clients)))
    return all_clients[['random']]


def load_features(
        conf,
        use_random=False,
):
    features = []
    if use_random:
        features.append(_random_features(conf))

    return features
