import logging
import os
from functools import partial

import numpy as np
import pandas as pd

import torch
import tqdm
from tqdm.autonotebook import tqdm

from dltranz.metric_learn.read_processing import fit_features, add_ticks, fit_types
from dltranz.data_load import read_dataset_mthread
from dltranz.util import get_data_files, get_conf
from dltranz.data_load import create_validation_loader, ConvertingTrxDataset, TrxDataset
from dltranz.train import score_model

logger = logging.getLogger(__name__)


class ModelEnsemble(torch.nn.Module):
    def __init__(self, submodels):
        super(ModelEnsemble, self).__init__()
        self.models = torch.nn.ModuleList(submodels)

    def forward(self, *args):
        out = []
        for m in self.models:
            o = m(*args)
            if len(o.size()) == 1:
                o = o.view(-1, 1)
            out.append(o)
        out = torch.cat(out, dim=1)
        return out


def load_model(conf):
    if 'model' in conf['model_path']:
        path = conf['model_path.model']
        model = torch.load(path)
        logger.info(f'Model loaded from "{path}"')
        return model
    elif 'models' in conf['model_path']:
        raise NotImplementedError()

        path = conf['model_path.models']
        models = [torch.load(p) for p in path]
        logger.info(f'Models loaded from "{path}"')
        return ModelEnsemble(models)

    elif 'state_dict' in conf['model_path']:
        raise NotImplementedError()
    else:
        raise AttributeError(f'Not supported model_path: {conf["model_path"]}')


def read_dataset_all(conf, desc, preproc_gen):
    logger.info(f'Data loading ({desc})...')
    files = get_data_files(conf['dataset'])
    n_workers = conf['dataset.n_workers']
    rec_gen = read_dataset_mthread(files, n_workers, preproc_gen)
    data = list(tqdm(rec_gen))
    logger.info(f'Loaded {len(data)} rows from disk ({desc})')
    return data


def infer_part_of_data(part_num, part_data, columns, model, conf, lock_obj=None):
    if lock_obj:
        lock_obj.acquire()

    if part_num is None:
        logger.info(f'Start to score data ({len(part_data)} records)')
    else:
        logger.info(f'Start to score {part_num} part of data ({len(part_data)} records)')

    if conf['dataset.preprocessing.add_seq_len'] and 'seq_len' not in columns:
        columns.append('seq_len')  # change list object

    valid_ds = ConvertingTrxDataset(TrxDataset(part_data))
    valid_loader = create_validation_loader(valid_ds, conf['params.valid'])

    _, pred = score_model(model, valid_loader, conf['params'])

    if conf['params.device'] != 'cpu':
        torch.cuda.empty_cache()
        logger.info('torch.cuda.empty_cache()')
    if lock_obj:
        lock_obj.release()

    if len(pred.shape) == 1:
        pred = pred.reshape(-1, 1)

    df_scores_cols = [f'v{i:003d}' for i in range(pred.shape[1])]
    df_scores = pd.DataFrame(pred, columns=df_scores_cols)

    df_labels = pd.DataFrame(({k: v for k, v in rec.items() if k in columns}
                              for rec in part_data))
    for col in df_labels:
        df_scores[col] = df_labels[col]
    df_scores = df_scores.reindex(columns=df_labels.columns.tolist() + df_scores_cols)
    logger.info(f'df_scores examples: {df_scores.shape}:')
    return df_scores

def save_scores(df_scores, part_num, output_conf):
    # output
    output_name = output_conf['path']
    output_format = output_conf['format']
    if output_format not in ('pickle', 'csv'):
        logger.warning(f'Format "{output_format}" is not supported. Used default "pickle"')
        output_format = 'pickle'

    if part_num is None:
        output_path = f'{output_name}.{output_format}'
    else:
        os.makedirs(output_conf['path'], exist_ok=True)
        output_path = f'{output_name}/{part_num:03}.{output_format}'

    if output_format == 'pickle':
        df_scores.to_pickle(output_path)
    elif output_format == 'csv':
        df_scores.to_csv(output_path, sep=',', header=True, index=False)
    else:
        raise AssertionError('Never happens')
    logger.info(f'{len(df_scores)} records saved to: "{output_path}"')


def score_part_of_data(part_num, part_data, columns, model, conf, lock_obj=None):
    df_scores = infer_part_of_data(part_num, part_data, columns, model, conf, lock_obj=lock_obj)

    save_scores(df_scores, part_num, conf['output'])


def common_preprocessing(seq, conf):
    preprocessing_conf = conf['dataset.preprocessing']

    fill_target = preprocessing_conf['fill_target']
    add_seq_len = preprocessing_conf['add_seq_len']
    min_date = np.datetime64(preprocessing_conf['min_date']) if 'min_date' in preprocessing_conf else None
    max_date = np.datetime64(preprocessing_conf['max_date']) if 'max_date' in preprocessing_conf else None

    for rec in seq:
        if 'application_date' in rec:
            application_date = np.datetime64(rec['application_date'])
            rec['application_date'] = application_date
            if min_date is not None and application_date < min_date:
                continue
            if max_date is not None and application_date >= max_date:
                continue

        if fill_target == 'empty':
            rec['target'] = -1

        if add_seq_len:
            tranz_dates = rec['event_time']
            n_tranz = len(tranz_dates)
            rec['seq_len'] = n_tranz

        yield rec


def consumer_preprocess_gen(seq, conf, data_pre_filter=None, ticks_mode='application_date'):
    if data_pre_filter is not None:
        seq = data_pre_filter(seq, conf)

    seq = common_preprocessing(seq, conf)
    seq = fit_features(seq,
                       embeddings=conf['params.trx_encoder.embeddings'],
                       numeric_values=conf['params.trx_encoder.numeric_values'])
    if 'tick_params' in conf['params']:
        seq = add_ticks(seq, conf, mode=ticks_mode)
    seq = fit_types(seq,
                    embeddings=conf['params.trx_encoder.embeddings'],
                    numeric_values=conf['params.trx_encoder.numeric_values'])
    return seq


def main_single_part(args=None):
    conf = get_conf(args)

    model = load_model(conf)
    columns = conf['output.columns']

    valid_data = read_dataset_all(conf, 'valid', partial(consumer_preprocess_gen, conf=conf))
    score_part_of_data(None, valid_data, columns, model, conf)
