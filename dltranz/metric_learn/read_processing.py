import numpy as np


def _get_dtype(max_size):
    if max_size <= 2 ** 8 - 1:
        return np.uint8, max_size
    if max_size <= 2 ** (16 - 1) - 1:
        return np.int16, max_size
    if max_size <= 2 ** (32 - 1) - 1:
        return np.int32, max_size
    return np.int64, max_size


def _dtype_mapping(embeddings, numeric_values):
    dtype_mapping = {k: _get_dtype(v['in']) for k, v in embeddings.items()}
    for k in numeric_values.keys():
        dtype_mapping[k] = np.int32, None
    return dtype_mapping


def _dtype_adjust(k, v, dtype_mapping):
    if k in dtype_mapping:
        dtype, max_value = dtype_mapping[k]
        if max_value is not None:
            v = np.where(
                v < max_value,
                v,
                1
            )
        return v.astype(dtype)

    if v.dtype == np.int8:
        return v.astype(np.int16)
    return v


def feature_local_day(feature_arrays, et):
    feature_arrays['local_day'] = ((et.astype('datetime64[D]') -
                                    et.astype('datetime64[M]')).astype(np.int16) + 1)


def feature_local_month(feature_arrays, et):
    feature_arrays['local_month'] = (et.astype('datetime64[M]').astype(np.int16) % 12 + 1)


def feature_local_weekday(feature_arrays, et):
    feature_arrays['local_weekday'] = ((et.astype('datetime64[D]').astype(np.int16) + 3) % 7 + 1)


def feature_hour(feature_arrays, et):
    feature_arrays['hour'] = ((et.astype('datetime64[s]') -
                               et.astype('datetime64[D]')).astype(np.int32) // 3600 + 1).astype(np.int16)


def feature_random_number(feature_arrays, et):
    feature_arrays['random_number'] = np.random.random(len(et))


def types_and_features(seq, embeddings, numeric_values):
    dtype_mapping = _dtype_mapping(embeddings, numeric_values)

    feature_list = []
    if 'local_day' in dtype_mapping:
        feature_list.append(feature_local_day)
    if 'local_month' in dtype_mapping:
        feature_list.append(feature_local_month)
    if 'local_weekday' in dtype_mapping:
        feature_list.append(feature_local_weekday)
    if 'hour' in dtype_mapping:
        feature_list.append(feature_hour)
    if 'random_number' in dtype_mapping:
        feature_list.append(feature_random_number)

    for rec in seq:
        et = rec['event_time']

        feature_arrays = rec['feature_arrays']

        # time features
        for f in feature_list:
            f(feature_arrays, et)

        # drop unused keys
        for k in set(feature_arrays) - set(dtype_mapping):
            del feature_arrays[k]

        # fill missing keys for empty records
        for k in dtype_mapping:
            if k not in feature_arrays:
                feature_arrays[k] = np.array([])

        # dtype_adjust
        feature_arrays = {k: _dtype_adjust(k, v, dtype_mapping) for k, v in feature_arrays.items()}

        rec['feature_arrays'] = feature_arrays
        yield rec


def fit_features(seq, embeddings, numeric_values):
    dtype_mapping = _dtype_mapping(embeddings, numeric_values)

    feature_list = []
    if 'local_day' in dtype_mapping:
        feature_list.append(feature_local_day)
    if 'local_month' in dtype_mapping:
        feature_list.append(feature_local_month)
    if 'local_weekday' in dtype_mapping:
        feature_list.append(feature_local_weekday)
    if 'hour' in dtype_mapping:
        feature_list.append(feature_hour)
    if 'random_number' in dtype_mapping:
        feature_list.append(feature_random_number)

    for rec in seq:
        et = rec['event_time']

        feature_arrays = rec['feature_arrays']

        # time features
        for f in feature_list:
            f(feature_arrays, et)

        # drop unused keys
        for k in set(feature_arrays) - set(dtype_mapping):
            del feature_arrays[k]

        # fill missing keys for empty records
        for k in dtype_mapping:
            if k not in feature_arrays:
                feature_arrays[k] = np.array([])

        rec['feature_arrays'] = feature_arrays
        yield rec


def fit_types(seq, embeddings, numeric_values):
    dtype_mapping = _dtype_mapping(embeddings, numeric_values)

    for rec in seq:
        feature_arrays = rec['feature_arrays']

        feature_arrays = {k: _dtype_adjust(k, v, dtype_mapping) for k, v in feature_arrays.items()}

        rec['feature_arrays'] = feature_arrays
        yield rec


def add_ticks(seq, conf, mode):
    assert mode in ('application_date', 'time_window')

    freq = conf['params.tick_params.freq']

    if mode == 'application_date':
        application_start_date = np.datetime64(conf['dataset.application_start_date'])
        transaction_start_date = np.datetime64(conf['dataset.transaction_start_date'])
        max_days = conf['params.max_days']
        trx_start_date = max(application_start_date - np.timedelta64(max_days, 'D'), transaction_start_date)
        trx_end_date = np.datetime64(conf['dataset.ripe_date'])
    elif mode == 'time_window':
        trx_start_date = np.datetime64(conf['dataset.start_date'])
        trx_end_date = np.datetime64(conf['dataset.end_date'])

    all_dates = np.arange(trx_start_date, trx_end_date, np.timedelta64(1, 'D'))
    if freq == 'w':
        weekdays = ((all_dates.astype('datetime64[D]').astype(np.int16) + 3) % 7 + 1)
        all_dates = all_dates[weekdays == 1]
    elif freq == 'm':
        monthdays = (all_dates.astype('datetime64[D]') - all_dates.astype('datetime64[M]') + 1).astype('int')
        all_dates = all_dates[monthdays == 15]
    else:
        raise AttributeError(f'Only "w" - week and "m" - month, freq supported. Found "{freq}"')

    all_zeros = np.zeros(len(all_dates), dtype=np.int16)

    for rec in seq:
        if mode == 'application_date':
            application_date = rec['application_date']

        et = rec['event_time']
        original_len = len(et)
        feature_arrays = rec['feature_arrays']

        if mode == 'application_date':
            add_dates = all_dates[all_dates <= application_date]
            add_dates = add_dates[add_dates >= application_date - np.timedelta64(max_days, 'D')]
            add_features = all_zeros[:len(add_dates)]
        elif mode == 'time_window':
            add_dates = all_dates
            add_features = all_zeros

        if len(et) > 0:
            et = np.concatenate([et, add_dates])
        else:
            et = add_dates
        ix_to_take = np.argsort(et)

        if 'tick' in feature_arrays:
            del feature_arrays['tick']
        feature_arrays = {k: np.concatenate([v, add_features]) for k, v in feature_arrays.items()}
        et = et[ix_to_take]
        feature_arrays = {k: v[ix_to_take] for k, v in feature_arrays.items()}
        feature_arrays['tick'] = (ix_to_take >= original_len).astype(np.int16)

        rec['feature_arrays'] = feature_arrays
        rec['event_time'] = et

        yield rec
