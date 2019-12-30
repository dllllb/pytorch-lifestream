import json
import logging
import os

import metric_learning_supervised_tuning
import metric_learning_train
import tools
from supervised_learning_train import create_model
from tools import logging_base_config, get_device
from tools.loss import ContrastiveLoss, HardNegativePairSelector, MSELossBalanced
from tools.model import CnnModel, freeze_layers, copy_model
from tools.params import save_model
from tools.shape_generator import ShapeGenerator
from tools.train import fit_model

logger = logging.getLogger(__name__)

num_processes = 1
train_batch_size = 128
max_epochs = 196
loss_margin = 0.2

lr = 0.0001  # was 0.0002
step_size = 8
gamma = 0.90

file_name_suffix = f'size-{lr}-{loss_margin}-{max_epochs}'

ml_params = {
    'train_path': "../datasets/metric_learning-8000-5-size.p",
    'valid_path': "../datasets/metric_learning-4000-5-size.p",

    'num_workers': 16,
    'train_batch_size': train_batch_size,
    'valid_batch_size': 128,
    'max_epochs': max_epochs,

    'class_params': ['size'],
    'ml_sample_count': 4,
    'ml_neg_count': 3,
    'loss_margin': loss_margin,

    'lr': lr,
    'step_size': step_size,
    'gamma': gamma,

    'history_path': f'../history/ml_vector_quality-{file_name_suffix}.json',
    'model_path_save_to': f'../models/ml-{file_name_suffix}.json',
}


sup_params = {
    'train_path': '../datasets/supervised-train-4000.p',
    'valid_path': '../datasets/supervised-valid-4000.p',

    'num_workers': 16,
    'train_batch_size': 64,
    'valid_batch_size': 64,

    'fit_params': {
        'num_processes': num_processes,
        'max_epochs': 40,
        'lr': 0.001,
        'step_size': 8,
        'gamma': 0.90,
    },
}


def get_epoch_step(current_epoch):
    epoch_step = 32

    if current_epoch < 256:
        epoch_step /= 2
    if current_epoch < 64:
        epoch_step /= 2

    return int(epoch_step)


def save_history(state, path):
    if os.path.isfile(path):
        with open(path, 'r') as f:
            history = json.load(f)
    else:
        history = []

    history.append(state)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    assert num_processes == 1, "Only single process mode supported"

    logging_base_config()
    device = get_device(num_processes)

    # metric learning
    ml_loaders = metric_learning_train.prepare_loaders(ml_params)
    tools.plot_data_loader_samples(8, ml_params['ml_sample_count'], ml_loaders['valid_loader'])

    ml_model = CnnModel(ShapeGenerator.image_shape)
    logger.info(f'Created new {ml_model.__class__.__name__}')

    ml_loss_fn = ContrastiveLoss(ml_params['loss_margin'],
                                 HardNegativePairSelector(ml_params['ml_neg_count']))

    # supervised tuning
    sup_loaders = metric_learning_supervised_tuning.prepare_loaders(sup_params)
    sup_loss_fn = MSELossBalanced()

    # main loop
    start_epoch = -1
    ml_score = {}
    while start_epoch < ml_params['max_epochs']:
        # metric learning iteration
        max_epochs_iteration = get_epoch_step(start_epoch)

        if start_epoch < 0:
            logger.info(f"Skip metric learning first iteration")
            start_epoch = -max_epochs_iteration
        else:
            logger.info(f"Start metric learning iteration from {start_epoch} epoch")

            fit_params = {
                'num_processes': 1,
                'max_epochs': max_epochs_iteration,
                'lr': lr * (gamma ** (start_epoch // step_size)),
                'step_size': step_size,
                'gamma': gamma,
            }

            ml_score = fit_model(
                seed=42,
                device=device,
                model=ml_model,
                loss_fn=ml_loss_fn,
                metrics_fn={},
                **ml_loaders,
                **fit_params,
            )

        # supervised train - evaluate
        logger.info(f"Start current vectors quality evaluation")

        _ml_model_cp = copy_model(ml_model)
        freeze_layers(_ml_model_cp)
        sup_model = create_model(_ml_model_cp)

        sup_score = fit_model(
            seed=42,
            device=device,
            model=sup_model,
            loss_fn=sup_loss_fn,
            metrics_fn={},
            **sup_loaders,
            **sup_params['fit_params'],
        )

        start_epoch += max_epochs_iteration
        history_state = {
            'ml_score': ml_score,
            'sup_score': sup_score,
            'current_epoch': start_epoch
        }
        save_history(history_state, ml_params['history_path'])

    save_model(ml_model, ml_score, sup_params=sup_params, **ml_params)
