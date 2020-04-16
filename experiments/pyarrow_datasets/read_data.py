import argparse
import datetime
import json
import os
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import pyarrow as pa
import pyarrow.parquet as pq
import sparkpickle


def read_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=os.path.abspath, required=True)
    parser.add_argument('--n_process', type=int, required=True)
    parser.add_argument('--cpu_count', type=int, default=0)
    parser.add_argument('--return_data', action='store_true')
    parser.add_argument('--engine', choices=['pyarrow', 'sparkpickle'], required=True)
    parser.add_argument('--output_file', default='results.json')

    args = parser.parse_args(args)
    return vars(args)


def read_pyarrow_file(path, use_threads, return_data):
    p_table = pq.read_table(
        source=path,
        use_threads=use_threads,
    )

    col_indexes = [n for n in p_table.column_names]

    def get_records():
        for rb in p_table.to_batches():
            col_arrays = [rb.column(i) for i, _ in enumerate(col_indexes)]
            col_arrays = [a.to_numpy(zero_copy_only=False) for a in col_arrays]
            for row in zip(*col_arrays):
                rec = {n: a for n, a in zip(col_indexes, row)}
                yield rec

    records = list(get_records())
    print('.', end='', flush=True)

    if return_data:
        return records
    else:
        return None


def read_sparkpickle_file(path_wc, return_data):
    def get_records():
        for path in glob(path_wc):
            with open(path, 'rb') as f:
                for rec in sparkpickle.load_gen(f):
                    yield rec

    records = list(get_records())
    print('.', end='', flush=True)

    if return_data:
        return records
    else:
        return None


def read_pyarrow(data_path, n_process, cpu_count, return_data, **kwargs):
    if cpu_count > 0:
        pa.set_cpu_count(cpu_count)

    if n_process == -1:
        data = read_pyarrow_file(data_path, cpu_count > 0, return_data)
    elif n_process == 0:
        data = []
        files = glob(data_path + '/part*')
        for p in files:
            data.append(read_pyarrow_file(p, cpu_count > 0, return_data))
    else:
        pool = Pool(n_process)
        data = list(pool.imap_unordered(partial(read_pyarrow_file, use_threads=cpu_count > 0, return_data=return_data),
                                        glob(data_path + '/part*')))

    if return_data:
        return data
    else:
        return None


def read_sparkpickle(data_path, n_process, return_data, **kwargs):
    if n_process == -1:
        data = read_sparkpickle_file(data_path + '/part*', return_data)
    elif n_process == 0:
        data = []
        files = glob(data_path + '/part*')
        for p in files:
            data.append(read_sparkpickle_file(p, return_data))
    else:
        pool = Pool(n_process)
        data = list(pool.imap_unordered(partial(read_sparkpickle_file, return_data=return_data),
                                        glob(data_path + '/part*')))

    if return_data:
        return data
    else:
        return None


if __name__ == '__main__':
    conf = read_args()
    str_conf = json.dumps(conf)
    print(f'conf: {str_conf}')

    _start = datetime.datetime.now()
    if conf['engine'] == 'pyarrow':
        read_pyarrow(**conf)
        print('')
    elif conf['engine'] == 'sparkpickle':
        read_sparkpickle(**conf)
        print('')
    else:
        raise NotImplementedError()

    _elapsed = datetime.datetime.now() - _start

    str_conf = json.dumps(conf)
    print(f'{_elapsed.seconds:6d} sec, {_elapsed}, {str_conf}')

    with open(conf['output_file'], 'a') as f:
        conf['elapsed'] = _elapsed.seconds
        str_conf = json.dumps(conf)
        f.write(str_conf + '\n')
