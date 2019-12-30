import logging


from torch.nn import Sequential, Linear, Sigmoid, BCELoss
from torch.utils.data.dataloader import DataLoader

from tools import logging_base_config, get_device, plot_data_loader_samples
from tools.data_loader import OnlyXDataset, MixedDataset
from tools.dataset import ImageDataset, load_dataset
from tools.dataset_transformers import ImageTransformer
from tools.loss import Accuracy
from tools.model import CnnModel, freeze_layers
from tools.params import load_model, save_model
from tools.shape_generator import ShapeGenerator
from tools.train import fit_model
from tools import dataset_transformers

logger = logging.getLogger(__name__)

num_processes = 1
max_epoch = 32  # 80

params = {
    'train_gen1_path': '../datasets/supervised-train_gen1-2000.p',
    'train_gen2_path': '../datasets/supervised-train_medium_size-2000.p',
    'valid_gen1_path': '../datasets/supervised-valid_gen1-2000.p',
    'valid_gen2_path': '../datasets/supervised-valid_medium_size-2000.p',

    'num_workers': 16,
    'train_batch_size': 64 // num_processes,
    'valid_batch_size': 64,

    'fit_params': {
        'num_processes': num_processes,
        'max_epochs': max_epoch // num_processes,
        'lr': 0.001,
        'step_size': 8 // num_processes,
        'gamma': 0.90,
    },

    # 'model_path_load_from': '../models/ml-color-0.0001-0.4-256.json',
    # 'model_freeze': True,  # True in main mode, False in debug mode

    'model_path_load_from': None,
    'model_freeze': False,  # True in main mode, False in debug mode

    'model_path_save_to': '../models/ml_discriminator.p',
}


if __name__ == '__main__':
    logging_base_config()
    device = get_device(num_processes)

    transformer1 = None
    transformer2 = None

    train_data = MixedDataset(
        ImageTransformer(OnlyXDataset(ImageDataset(load_dataset(params['train_gen1_path']))), transformer1),
        ImageTransformer(OnlyXDataset(ImageDataset(load_dataset(params['train_gen2_path']))), transformer2),
    )
    train_loader = DataLoader(train_data, batch_size=params['train_batch_size'],
                              shuffle=True, num_workers=params['num_workers'])

    valid_data = MixedDataset(
        ImageTransformer(OnlyXDataset(ImageDataset(load_dataset(params['valid_gen1_path']))), transformer1),
        ImageTransformer(OnlyXDataset(ImageDataset(load_dataset(params['valid_gen2_path']))), transformer2),
    )
    valid_loader = DataLoader(valid_data, batch_size=params['train_batch_size'],
                              shuffle=False, num_workers=params['num_workers'])

    plot_data_loader_samples(8, 8, valid_loader)

    ml_model = load_model(**params)
    if ml_model is None:
        ml_model = CnnModel(ShapeGenerator.image_shape)
        logger.info(f'New {ml_model.__class__.__name__} created')
    if params['model_freeze']:
        freeze_layers(ml_model)

    discriminator = Sequential(
        Linear(ml_model.vector_size, 1),
        Sigmoid(),
    )

    model = Sequential(
        ml_model,
        discriminator,
    )

    loss_fn = BCELoss()
    metrics_fn = {'accuracy': Accuracy()}

    score = fit_model(
        seed=42,
        device=device,
        model=model,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        **params['fit_params'],
    )

    save_model(ml_model, score, **params)
