import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import os
import json

import dltranz.neural_automl.lib as lib
import dltranz.neural_automl.neural_automl_models as mdls


'''
X_train, X_valid - numpy arrays with shape [n_samples, n_features]
y_train, y_valid - targets [n_samples]
config - if None then using params from conf/default_config.json
'''
def train_from_config(X_train, y_train, X_valid, y_valid, config=None):
    #
    def_conf = 'conf/binary_classification_config.json'
    if config is not None and isinstance(config, str):
        def_conf = 'conf/' + config
        if not osp.exists(def_conf):
            def_conf = 'conf/binary_classification_config.json'
            print(f"config file {'conf/' + config} not exists, using {def_conf} instead")
    if config is None or not isinstance(config, dict) or not config.get('layers', False):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), def_conf), 'r') as f:
            config = json.load(f)
        print(f"\nModel config is not specified. Using default one instead (see neural_automl/conf/default_config.json):")
        #print(json.dumps(config, indent=4))

    # loading data
    print(f"\nPreprocess data| normalize={config['data_transform_params']['normalize']}, quantile_transform={config['data_transform_params']['quantile_transform']}, quantile_noise={config['data_transform_params']['quantile_noise']}")
    data = lib.Dataset(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
                       random_state=None,  #1337,
                       normalize=config['data_transform_params']['normalize'],
                       quantile_transform=config['data_transform_params']['quantile_transform'],
                       quantile_noise=config['data_transform_params']['quantile_noise'])

    #
    model = mdls.get_model_from_config(config, data.X_train)

    # loss
    if config['loss_params']['func'] == 'binary_cross_entropy_with_logits':
        loss_function = F.binary_cross_entropy_with_logits
    elif config['loss_params']['func'] == 'binary_cross_entropy':
        loss_function = F.binary_cross_entropy
    elif config['loss_params']['func'] == 'CrossEntropyLoss':
        loss_function = nn.CrossEntropyLoss()
    elif config['loss_params']['func'] == 'MSELoss':
        loss_function = nn.torch.nn.MSELoss()
    else:
        raise NotImplemented(f"unknown loss function type {config['loss_params']['func']}")

    #
    lr = config['sgd_params']['lr']
    trainer = lib.Trainer(
        model=model,
        loss_function=loss_function,
        experiment_name=None,
        warm_start=False,
        Optimizer=Adam,
        optimizer_params={'lr': lr},
        verbose=False,
        n_last_checkpoints=5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.opt, step_size=config['sgd_params']['step_size'], gamma=config['sgd_params']['step_gamma'])

    loss_history, err_history = [], []
    for epoch in range(config['train_params']['n_epoch']):
        _train(trainer, config, data, epoch + 1, loss_history)
        _test(trainer, config, data, epoch + 1, err_history)
        scheduler.step()
    return np.max(err_history)


def _train(trainer, config, data, epoch, loss_history):
    with tqdm.tqdm(total=int(data.y_train.size/config['train_params']['batch_size'])) as steps:
        for batch in lib.iterate_minibatches(data.X_train, data.y_train, batch_size=config['train_params']['batch_size'],
                                             shuffle=True, epochs=1):
            metrics = trainer.train_on_batch(*batch, device=config['device'])
            loss_history.append(metrics['loss'].item())
            loss = np.mean(loss_history)

            steps.set_description(f"train: epoch {epoch}/{config['train_params']['n_epoch']}")
            steps.set_postfix({'loss': '{:.5E}'.format(loss)})
            steps.update()


def _test(trainer, config, data, epoch, err_history):
    trainer.save_checkpoint()
    trainer.average_checkpoints(out_tag='avg')
    trainer.load_checkpoint(tag='avg')
    roc_auc = trainer.evaluate_binary_auc(data.X_valid, data.y_valid, device=config['device'], batch_size=config['valid_params']['batch_size'])
    err_history.append(roc_auc)

    if roc_auc >= np.max(err_history):
        trainer.save_checkpoint(tag='best')

    trainer.load_checkpoint()  # last
    trainer.remove_old_temp_checkpoints()
    print(f'epoch {epoch}, ROC_AUC: {roc_auc}')