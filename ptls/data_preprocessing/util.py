import numpy as np
import pandas as pd


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


def update_with_target(features, df_target, col_id, col_target):
    d_clients = df_target.to_dict(orient='index')
    features = [
        dict([('target', d_clients.get(rec[col_id], {}).get(col_target))] + list(rec.items()))
        for rec in features
    ]
    return features
