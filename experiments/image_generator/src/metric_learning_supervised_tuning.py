"""
Ожидается, что при этом он неявно выучит завиксированный параметр
То есть ML вектор сможет предсказывать его
"""
import logging

from torch.nn import Sequential, Linear, ReLU
from torch.utils.data.dataloader import DataLoader

from supervised_learning_train import create_model
from tools import get_device, logging_base_config
from tools.dataset import ImageDataset, load_dataset
from tools.loss import MSELossBalanced
from tools.model import freeze_layers, CnnModel
from tools.params import load_model, save_model
from tools.shape_generator import ShapeGenerator
from tools.train import fit_model

logger = logging.getLogger(__name__)

num_processes = 1
max_epoch = 32  # 80

params = {
    'train_path': '../datasets/supervised-train-4000.p',
    'valid_path': '../datasets/supervised-valid-4000.p',

    'num_workers': 16,
    'train_batch_size': 64 // num_processes,
    'valid_batch_size': 64,

    'fit_params': {
        'num_processes': num_processes,
        'max_epochs': max_epoch // num_processes,
        'lr': 0.002,
        'step_size': 8 // num_processes,
        'gamma': 0.90,
    },

    'model_path_load_from': '../models/ml_discriminator.p',
    'model_path_save_to': f'../models/ml_discriminator-tuning-{max_epoch}.p',
}


def prepare_loaders(params):
    train_data = ImageDataset(load_dataset(params['train_path']))
    valid_data = ImageDataset(load_dataset(params['valid_path']))

    train_loader = DataLoader(train_data, batch_size=params['train_batch_size'],
                              shuffle=True, num_workers=params['num_workers'])
    valid_loader = DataLoader(valid_data, batch_size=params['valid_batch_size'],
                              shuffle=False, num_workers=params['num_workers'])

    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
    }


if __name__ == '__main__':
    logging_base_config()
    device = get_device(num_processes)

    loaders = prepare_loaders(params)

    ml_model = load_model(**params)
    if ml_model is None:
        ml_model = CnnModel(ShapeGenerator.image_shape)
        logger.info(f'New {ml_model.__class__.__name__} created')
    freeze_layers(ml_model)
    model = create_model(Sequential(ml_model, Linear(64, 64), ReLU()))

    loss_fn = MSELossBalanced()

    score = fit_model(
        seed=42,
        device=device,
        model=model,
        metrics_fn={},
        loss_fn=loss_fn,
        **loaders,
        **params['fit_params'],
    )

    save_model(model, score, **params)
