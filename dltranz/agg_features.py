"""
This module prepare baseline features from standard dataset.
`Standard dataset` means iterable with dictionaries:
    {
        <id_col_name>:  <id_value>,
        <label_x_name>: <label_x_value>,
        'event_time': np.array([...]),
        'feature_arrays': {
            <feature_1_name>: np.array([...]),
            <feature_2_name>: np.array([...]),
            ...,
            <feature_n_name>: np.array([...]),
        },
    }

It's possible use such format for NN train (MLES, or supervised training or anywhere) or inference
with `dltranz` module.

Also you can convert `feature_arrays` to "baseline features" - aggregates over categorical features
with this module.

`make_trx_features` convert standard dataset (see above) to flat aggregates:
    {
        <id_col_name>:        <id_value>,
        <label_x_name>:       <label_x_value>,
        <agg_feature_1_name>: <agg_feature_1_value>,
        <agg_feature_2_name>: <agg_feature_2_value>,
        ...,
        <agg_feature_m_name>: <agg_feature_m_value>,
    }
where <agg_feature_1_name> - <agg_feature_m_name> can be obtained via `get_feature_names`

Prepared features:
  - haven't NA values
  - requires normalization

"""


import numpy as np


_NUM_PERCENTILE_LIST = [0, 10, 25, 50, 75, 90, 100]


def _num_features(col, val):
    val_orig = val
    return {
        f'{col}_count': len(val),
        f'{col}_sum': val_orig.sum(),
        f'{col}_std': val_orig.std(),
        f'{col}_mean': val.mean(),
        **{f'{col}_p{p_level}': val
           for val, p_level in zip(np.percentile(val, _NUM_PERCENTILE_LIST), _NUM_PERCENTILE_LIST)}
    }


def _embed_features(col_embed):
    return {
        f'{col_embed}_nunique': np.unique(col_embed).size,
    }


def _cross_features(col_embed, col_num, val_embed, val_num, size):
    def norm_row(a, agg_func):
        return a / (agg_func(a) + 1e-5)

    val_num_orig = val_num

    seq_len = len(val_embed)

    m_cnt = np.zeros((seq_len, size), np.float)
    m_sum = np.zeros((seq_len, size), np.float)
    ix = np.arange(seq_len)

    m_cnt[(ix, val_embed)] = 1
    m_sum[(ix, val_embed)] = val_num_orig

    return {
        **{f'{col_embed}_X_{col_num}_{k}_cnt': v for k, v in enumerate(norm_row(m_cnt.sum(axis=0), np.sum))},
        **{f'{col_embed}_X_{col_num}_{k}_sum': v for k, v in enumerate(norm_row(m_sum.sum(axis=0), np.sum))},
        **{f'{col_embed}_X_{col_num}_{k}_std': v for k, v in enumerate(norm_row(m_sum.std(axis=0), np.sum))},
    }


def make_trx_features(seq, conf):
    numeric_values = conf['params.trx_encoder.numeric_values']
    embeddings = conf['params.trx_encoder.embeddings']

    for rec in seq:
        # labels and target
        features = {k: v for k, v in rec.items() if k not in ('event_time', 'feature_arrays')}
        event_time = rec['event_time']
        feature_arrays = rec['feature_arrays']

        for col_num, options_num in numeric_values.items():
            # take array with numerical feature and convert it to original scale
            val_orig = feature_arrays[col_num]
            if options_num.get('was_logified', False):
                val_orig = np.expm1(abs(val_orig)) * np.sign(val_orig)

            # trans_common_features
            features.update(_num_features(col_num, val_orig))

            # embeddings features (like mcc)
            for col_embed, options_embed in embeddings.items():
                features.update(_cross_features(
                    col_embed, col_num,
                    feature_arrays[col_embed], val_orig,
                    options_embed['in'])
                )

        for col_embed in embeddings.keys():
            features.update(_embed_features(col_embed))

        yield features


def get_feature_names(conf):
    numeric_values = conf['params.trx_encoder.numeric_values']
    embeddings = conf['params.trx_encoder.embeddings']

    # labels and target
    feature_col_names = []

    # trans_common_features
    for col_num in numeric_values.keys():
        feature_col_names.extend([
            f'{col_num}_count',
            f'{col_num}_sum',
            f'{col_num}_std',
            f'{col_num}_mean',
            *[f'{col_num}_p{p_level}' for p_level in _NUM_PERCENTILE_LIST]
        ])

        # embeddings features (like mcc)
        for col_embed, options in embeddings.items():
            feature_col_names.extend([
                *[f'{col_embed}_X_{col_num}_{k}_cnt' for k in range(options['in'])],
                *[f'{col_embed}_X_{col_num}_{k}_sum' for k in range(options['in'])],
                *[f'{col_embed}_X_{col_num}_{k}_std' for k in range(options['in'])],
            ])

    for col_embed in embeddings.keys():
        feature_col_names.extend([
            f'{col_embed}_nunique',
        ])

    return feature_col_names
