import matplotlib.pyplot as plt
from torch.nn import Sequential, Linear, Sigmoid
from torch.utils.data.dataloader import DataLoader

from tools import get_device, logging_base_config
from tools.dataset import load_dataset, ImageDataset
from tools.loss import MSELossBalanced
from tools.model import CnnModel, MultiHead
from tools.params import load_model, save_model
from tools.shape_generator import ShapeGenerator
from tools.train import fit_model

num_processes = 1
max_epoch = 80

params = {
    'train_path': '../datasets/supervised-train-4000.p',
    'valid_path': '../datasets/supervised-valid-4000.p',

    'num_workers': 16,
    'train_batch_size': 64 // num_processes,
    'valid_batch_size': 64,

    'fit_params': {
        'num_processes': num_processes,
        'max_epochs': max_epoch // num_processes,  # 40
        'lr': 0.0005,
        'step_size': 8 // num_processes,
        'gamma': 0.90,
    },

    'model_path_load_from': None,
    'model_path_save_to': f'../models/supervised-{max_epoch}.p',
}

generator_params = {
    'keep_params': [],
}


def plot_samples(n_obj, n_samples, shape_params):
    sgs = [ShapeGenerator(**shape_params) for _ in range(n_obj)]
    figire, axs = plt.subplots(n_obj, n_samples, figsize=(2 * n_samples, 2 * n_obj), dpi=74)
    for col in range(n_samples):
        for row in range(n_obj):
            ax = axs[row, col]
            ax.axis('off')
            ax.imshow(sgs[row].get_buffer()[0])
    plt.show()


def create_model(cnn_model=None):
    if cnn_model is None:
        cnn_model = CnnModel(ShapeGenerator.image_shape)

    model = Sequential(
        cnn_model,
        MultiHead({k: Sequential(Linear(CnnModel.vector_size, v), Sigmoid())
                   for k, v in ShapeGenerator.params_pattern.items()}),
    )
    return model


if __name__ == '__main__':
    logging_base_config()
    device = get_device(num_processes)

    model = load_model(**params)
    if model is None:
        model = create_model()

    train_data = ImageDataset(load_dataset(params['train_path']))
    valid_data = ImageDataset(load_dataset(params['valid_path']))

    train_loader = DataLoader(train_data, batch_size=params['train_batch_size'],
                              shuffle=True, num_workers=params['num_workers'])
    valid_loader = DataLoader(valid_data, batch_size=params['valid_batch_size'],
                              shuffle=False, num_workers=params['num_workers'])

    loss_fn = MSELossBalanced()

    score = fit_model(
        seed=42,
        device=device,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        **params['fit_params'],
    )

    save_model(model, score, **params)
