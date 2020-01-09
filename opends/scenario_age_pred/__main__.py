import argparse
import logging
import os

import pandas as pd
from multiprocessing import Pool

import xgboost as xgb
from sklearn.model_selection import train_test_split

from scenario_age_pred.features import load_features

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--data_path', type=os.path.abspath, default='data/age-pred/')
    parser.add_argument('--test_size', type=float, default=0.4)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--model_seed', type=int, default=42)

    config = parser.parse_args(args)
    return vars(config)


def read_target(conf):
    train_target = pd.read_csv(os.path.join(conf['data_path'], 'train_target.csv')).set_index('client_id')
    train_target, valid_target = train_test_split(
        train_target, test_size=conf['test_size'], stratify=train_target['bins'], random_state=conf['random_state'])
    return train_target, valid_target


def get_scores(args):
    pos, conf, params = args

    train_target, valid_target = read_target(conf)
    features = load_features(conf, **params)

    y_train = train_target['bins']
    y_valid = valid_target['bins']

    X_train = pd.concat([df.reindex(index=train_target.index) for df in features])
    X_valid = pd.concat([df.reindex(index=valid_target.index) for df in features])

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        n_jobs=4,
        seed=conf['model_seed'],
        n_estimators=300)

    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    accuracy = (y_valid == pred).mean()

    res = params.copy()
    res['pos'] = pos
    res['accuracy'] = accuracy
    return res


if __name__ == '__main__':
    conf = parse_args(None)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    param_list = [
        {'use_random': True}
    ]
    args_list = [(i, conf, params) for i, params in enumerate(param_list)]

    pool = Pool(processes=conf['n_workers'])
    results = pool.map(get_scores, args_list)
    df_results = pd.DataFrame(results).set_index('pos').sort_index()

    logger.info(f'Results:\n{df_results}')
