"""
Фиксируем у объектов одного класса один из параметров
Учим metric_learning так, чтобы он отличал объекты разных классов
Ожидается, что при этом он неявно выучит завиксированный параметр
То есть ML вектор сможет предсказывать его
"""

from torch.utils.data.dataloader import DataLoader

from tools import get_device, logging_base_config, plot_data_loader_samples
from tools.data_loader import collate_list_fn
from tools.dataset import ImageMetricLearningDatasetOnline, load_dataset, ImageMetricLearningDataset
from tools.loss import ContrastiveLoss, HardNegativePairSelector
from tools.model import CnnModel
from tools.params import load_model, save_model
from tools.shape_generator import ShapeGenerator
from tools.train import fit_model

num_processes = 1
max_epochs = 128  # 480
train_batch_size = 128
params = {
    'train_size': 4000,
    'valid_path': "../datasets/metric_learning-4000-4-bg_color_n_vertex_size.p",

    'num_workers': 16,
    'train_batch_size': train_batch_size,
    'valid_batch_size': 128,

    'class_params': ['size', 'n_vertex', 'bg_color'],
    'ml_sample_count': 4,
    'ml_neg_count': 3,
    'loss_margin': 0.4,

    'fit_params': {
        'num_processes': num_processes,
        'max_epochs': max_epochs // num_processes,  # 48
        'lr': 0.0002 * num_processes,
        'step_size': 8 // num_processes,
        'gamma': 0.95,
    },

    'model_path_load_from': None,
    'model_path_save_to': f'../models/ml-4000-5-except_color-{max_epochs}-{train_batch_size}.p',
}


def prepare_loaders(params):
    if 'train_size' in params:
        train_data = ImageMetricLearningDatasetOnline([ShapeGenerator(keep_params=params['class_params'])
                                                       for _ in range(params['train_size'])], params['ml_sample_count'])
    elif 'train_path' in params:
        train_data = ImageMetricLearningDataset(load_dataset(params['train_path']), params['ml_sample_count'])
    else:
        raise AttributeError('There is no information about train dataset')

    valid_data = ImageMetricLearningDataset(load_dataset(params['valid_path']), params['ml_sample_count'])

    train_loader = DataLoader(train_data, batch_size=params['train_batch_size'],
                              shuffle=True, num_workers=params['num_workers'],
                              collate_fn=collate_list_fn)
    valid_loader = DataLoader(valid_data, batch_size=params['valid_batch_size'],
                              shuffle=False, num_workers=params['num_workers'],
                              collate_fn=collate_list_fn)

    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
    }


if __name__ == '__main__':
    logging_base_config()
    device = get_device(num_processes)

    loaders = prepare_loaders(params)
    model = load_model(**params)
    if model is None:
        model = CnnModel(ShapeGenerator.image_shape)
        print(f'Created new {model.__class__.__name__}')

    loss_fn = ContrastiveLoss(params['loss_margin'],
                              HardNegativePairSelector(params['ml_neg_count']))

    # samples
    plot_data_loader_samples(8, params['ml_sample_count'], loaders['valid_loader'])

    score = fit_model(
        seed=42,
        device=device,
        model=model,
        loss_fn=loss_fn,
        metrics_fn={},
        **loaders,
        **params['fit_params']
    )

    save_model(model, score, **params)
