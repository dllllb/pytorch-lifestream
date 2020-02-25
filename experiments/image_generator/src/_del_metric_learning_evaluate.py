"""
Ожидается, что при этом он неявно выучит завиксированный параметр
То есть ML вектор сможет предсказывать его
"""
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from torch.utils.data.dataloader import DataLoader

from tools import ShapeGenerator, ImageMetricLearningDataset, ImageDataset, get_device
from tools.data_loader import collate_list_fn
from tools.model import CnnModel, freeze_layers
from tools.loss import ContrastiveLoss, HardNegativePairSelector
from tools.params import load_model
from tools.train import fit_model, valid

all_params = ['color', 'bg_color', 'size', 'n_vertex']
class_params = ['color']
print(f'class_param: {class_params}')

params = {
    'train_size': 4000,
    'batch_size': 32,
    'num_workers': 1,
    'num_processes': 5,
    'max_epoch': 50,

    'model_path_load_from': '../models/ml-model-100.p',
    'model_path_save_to': 'models/ml-100-scoring.p',
}

#
device = get_device()

ml_model = load_model(**params)
freeze_layers(ml_model)


results = {}
print('Train')
for target_name, out_size, max_epoch, lr, gamma in zip(
        ['color', 'bg_color', 'size', 'n_vertex'],
        [3, 3, 1, 1],
        [4, 4, 4, 4],  # [5, 5, 5, 5],
        [0.005, 0.005, 0.005, 0.005],
        [0.95, 0.95, 0.85, 0.85]
):
    if target_name not in ('color',):
        pass
        # continue

    print(f'\n{target_name}:')
    train_data = ImageDataset([ShapeGenerator(keep_params=[])
                               for _ in range(train_size)], target_name=target_name)
    valid_data = ImageDataset([ShapeGenerator(keep_params=[])
                               for _ in range(500)], target_name=target_name)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    loss_fn = MSELoss()
    model = torch.nn.Sequential(
        ml_model,
        torch.nn.Linear(64, out_size),
        torch.nn.Sigmoid(),
    )

    score = fit_model(
        seed=42,
        num_processes=num_processes,
        max_epochs=max_epoch,
        device=device,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=lr,
        step_size=5 // num_processes,
        gamma=gamma,
        loss_fn=loss_fn,
    )

    print(f'RESULTS: {target_name} : {score}')
    results[target_name] = {
        'model': model,
        'score': score,
    }

print('Evaluate')
for target_name, out_size in zip(
        ['color', 'bg_color', 'size', 'n_vertex'],
        [3, 3, 1, 1],
):
    print(f'\n{target_name}:')
    valid_data = ImageDataset([ShapeGenerator(keep_params=[])
                               for _ in range(500)], target_name=target_name)

    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = results.get(target_name)
    if model is not None:
        model = model['model']
    else:
        model = CnnModel(ShapeGenerator.image_shape, out_size)
        results[target_name] = {'model': model}

    score = valid(
        seed=42,
        test_loader=valid_loader,
        device=device,
        model=model,
        loss_fn=MSELoss(),
    )

    print(f'RESULTS: {target_name} : {score}')


print('')
for k, m in results.items():
    print('{:20s}: {}'.format(k, m.get('score')))


# Evaluate
n_obj = 12
n_sample = 3

fig, axs = plt.subplots(n_obj, n_sample * 2 + 2, figsize=(2 * (n_sample * 2 + 2), 2 * n_obj))

for m in results.values():
    m['model'].eval()

for row in range(n_obj):
    shape = ShapeGenerator(keep_params=['size', 'n_vertex', 'color', 'bg_color'])
    for col in range(n_sample):
        ax = axs[row, col]
        ax.axis('off')
        ax.imshow(shape.get_buffer()[0])

    X, y = shape.get_buffer()
    ax = axs[row, n_sample]
    ax.axis('off')
    ax.text(0, 0, '\n'.join(['{}: [{}]'.format(k, ', '.join([f'{i:.2f}' for i in v]))
                             for k, v in sorted(y.items())]))

    x = torch.from_numpy(X.reshape((1,) + X.shape))
    x = x.to(device)
    with torch.no_grad():
        out = model(x).cpu().numpy()
        pred_params = {
            'color': results['color']['model'](x).cpu().numpy()[0],
            'bg_color': results['bg_color']['model'](x).cpu().numpy()[0],
            'size': results['size']['model'](x).cpu().numpy()[0],
            'n_vertex': results['n_vertex']['model'](x).cpu().numpy()[0],
        }
    pred_shape = ShapeGenerator(keep_params=[], params=pred_params)

    for col in range(n_sample):
        ax = axs[row, n_sample + 1 + col]
        ax.axis('off')
        ax.imshow(pred_shape.get_buffer()[0])

    ax = axs[row, n_sample * 2 + 1]
    ax.axis('off')
    ax.text(0, 0, '\n'.join(['{}: [{}]'.format(k, ', '.join([f'{i:.2f}' for i in v]))
                             for k, v in sorted(pred_params.items())]))

plt.show()
