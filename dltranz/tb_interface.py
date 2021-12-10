import os
from glob import glob

import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm


def get_versions(path):
    version_list = glob(os.path.join(path, 'version_*'))
    for version_path in version_list:
        yield version_path, os.path.split(version_path)[-1]


def get_scalars(path):
    df_list = []
    for version_path, version_name in get_versions(path):
        events_path = glob(os.path.join(version_path, 'events.*'))[0]
        event_acc = EventAccumulator(events_path)
        event_acc.Reload()

        df_list.append(pd.DataFrame((
            (version_name, tag, e.step, e.value)
            for tag in event_acc.Tags()['scalars']
            for e in event_acc.Scalars(tag)
        ), columns=['version', 'tag', 'step', 'value']))
    return pd.concat(df_list, axis=0)


def get_last_scalars(path):
    df = get_scalars(path)
    df = df[lambda x: x['step'].gt(0)]
    df = df.set_index(['version', 'tag', 'step']).sort_index()
    df = df.groupby(['version', 'tag'])['value'].last().unstack()

    return df


def get_hparams(path, progress=False):
    df_list = []
    for version_path, version_name in tqdm(get_versions(path), disable=not progress):
        hp_path = glob(os.path.join(version_path, 'hparams.yaml'))[0]
        with open(hp_path, 'r') as f:
            hp = yaml.load(f, Loader=yaml.UnsafeLoader)
            hp = pd.DataFrame([hp], index=[version_name])
            df_list.append(hp)
    return pd.concat(df_list, axis=0)
