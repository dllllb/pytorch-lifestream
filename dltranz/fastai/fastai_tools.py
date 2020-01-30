from fastai.tabular import *
import pandas as pd
import numpy as np
import os
import os.path as osp
import json
import sys
sys.path.insert(0, '/mnt/data/molchanov/dltranz')
import dltranz.neural_automl.neural_automl_models as node
import torch.nn as nn

class TabularModelWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, *args):
        return self.m(args[1])

def train(X_train, y_train, X_valid, y_valid):
    data = to_fastai_data(X_train, y_train, X_valid, y_valid)

    learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
    learn.fit_one_cycle(10, 1e-2)
    return 0.56


def train_from_config(X_train, y_train, X_valid, y_valid, config):
    # model
    if config is None:
        raise NotImplementedError('you should specify config file')

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    with open(osp.join(base_path, 'conf', config), 'r') as f:
        model_config = json.load(f)
    model = node.get_model(model_config, X_train)
    model = TabularModelWrapper(model)

    # loss
    if model_config['loss_params']['func'] == 'binary_cross_entropy_with_logits':
        loss_function = F.binary_cross_entropy_with_logits
    elif model_config['loss_params']['func'] == 'binary_cross_entropy':
        loss_function = F.binary_cross_entropy
    elif model_config['loss_params']['func'] == 'cross_entropy':
        loss_function = nn.CrossEntropyLoss()
    elif model_config['loss_params']['func'] == 'nll':
        loss_function = nn.NLLLoss()
    elif model_config['loss_params']['func'] == 'mse':
        loss_function = nn.MSELoss()
    else:
        raise NotImplementedError(f"unknown loss function type {config['loss_params']['func']}")

    # metrics
    test_type = model_config['accuracy_params']['func']
    if test_type == 'roc_auc':
        metric = auc_roc_score
    elif test_type == 'accuracy':
        metric = accuracy
    elif test_type == 'mse':
        metric = root_mean_squared_error
    else:
        raise NotImplementedError(f"unknown test {test_type} in accuracy params")

    # train
    data = to_fastai_data(X_train, y_train, X_valid, y_valid)
    learner = Learner(data, 
                      model, 
                      loss_func=loss_function, 
                      metrics=metric)
    learner.fit_one_cycle(model_config["train_params"]["n_epoch"], 0.004)#1e-2)
    return -1


def to_fastai_data(X_train, y_train, X_valid, y_valid):
    data = np.concatenate((
                    np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1),
                    np.concatenate((X_valid, y_valid.reshape(-1, 1)), axis=1)
                    ), axis=0)
    cols = [f'col_{i}' for i in range(data.shape[-1])]
    cols[-1] = 'target'
    df = pd.DataFrame(data=data, columns=cols)
    df['target'] = df['target'].astype(y_train.dtype)

    # train test split
    len_test = y_valid.shape[0]
    len_train = y_train.shape[0]
    valid_idx = range(len_train, len_train + len_test)

    preprocess = []
    target = 'target'
    cat_vars = []
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    return TabularDataBunch.from_df(path, df, target, valid_idx=valid_idx, procs=preprocess, cat_names=cat_vars)

#train_from_config(np.random.randn(100, 5),np.random.randn(100, 5),np.random.randn(100, 5),np.random.randn(100, 5), 'age.json')