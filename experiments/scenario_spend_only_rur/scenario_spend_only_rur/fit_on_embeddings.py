import logging
import sys

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

from dltranz.experiment import update_model_stats
from dltranz.metric_learn.inference_tools import infer_part_of_data, save_scores
from dltranz.trx_encoder import TrxEncoder
from dltranz.util import init_logger, get_conf
from scenario_spend_only_rur.fit_target import create_ds, run_experiment, read_consumer_data, read_embedding_data

logger = logging.getLogger(__name__)

class EmbTranslation(torch.nn.Module):
    def __init__(self):
        super(EmbTranslation, self).__init__()
    def forward(self, x):
        return x.payload['embedding']


def load_model(conf):
    model_type = conf['model_type']
    if model_type == 'rnn':
        input_size = conf['rnn.hidden_size']
    elif model_type == 'transf':
        input_size = conf['transf.input_size']
    else:
        raise NotImplementedError(f'NotImplementedError for model_type="{model_type}"')

    head_output_size = 53

    layers = [EmbTranslation()]

    #if conf.get('freeze_layers', False):
    #    for i,_ in enumerate(layers):
    #       for p in layers[i].parameters():
    #            p.requires_grad = False

    if conf['use_batch_norm']:
        layers.append(torch.nn.BatchNorm1d(input_size))

    big_extension = [torch.nn.Linear(input_size, int(0.5*input_size))]+\
                    [torch.nn.ReLU(), torch.nn.BatchNorm1d(int(0.5*input_size)) ]+\
                    [torch.nn.Linear(int(0.5*input_size), 2*head_output_size) ]+\
                    [torch.nn.ReLU(), torch.nn.BatchNorm1d(2*head_output_size)]+\
                    [torch.nn.Linear(2*head_output_size, head_output_size) ]

    if conf.get('big_extension', False):
        layers.extend(big_extension)
    else:
        layers.extend([torch.nn.Linear(input_size, head_output_size) ])

    model = torch.nn.Sequential(*layers)
    return model

#is necessary for __main__
def prepare_parser(parser):
    pass

def main(_):
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('metric_learning')

    conf = get_conf(sys.argv[2:])

    model_f = load_model
    if conf['params'].get('embeddings_path', False):
      train_data = read_embedding_data(conf['params']['embeddings_path'], conf['dataset.train_path'] , conf)
      test_data = read_embedding_data(conf['params']['embeddings_path'], conf['dataset.test_path'] , conf)
    else:
      raise NotImplementedError(f'NotImplementedError. Set embeddings_path in config file!')
    
    # train
    results = []

    skf = StratifiedKFold(conf['cv_n_split'])
    nrows = conf['params'].get('labeled_amount',-1) # semi-supervised setup. default = supervised

    target_values = [rec['target'] for rec in train_data]
    for i, (i_train, i_valid) in enumerate(skf.split(train_data, np.zeros(shape=(len(train_data),1)) )):#target_values)):
        logger.info(f'Train fold: {i}')
        i_train_data = [rec for i, rec in enumerate(train_data) if i in i_train]
        i_valid_data = [rec for i, rec in enumerate(train_data) if i in i_valid]

        if nrows > 0: i_train_data = i_train_data[:nrows]

        train_ds, valid_ds = create_ds(i_train_data, i_valid_data, conf)
        model, result = run_experiment(train_ds, valid_ds, conf['params'], model_f)

        # inference
        columns = conf['output.columns']
        train_scores = infer_part_of_data(i, i_valid_data, columns, model, conf)
        save_scores(train_scores, i, conf['output.valid'])

        test_scores = infer_part_of_data(i, test_data, columns, model, conf)
        save_scores(test_scores, i, conf['output.test'])

        results.append(result)

    # results
    stats_file = conf.get('stats.path', None)
    if stats_file is not None:
        update_model_stats(stats_file, conf, results)
