import logging

import matplotlib.pyplot as plt
import torch
from matplotlib import pyplot as plt


def logging_base_config():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s')


def plot_xy(x, y):
    n_samples = 4
    n_obj = x.size()[0] // n_samples

    figire, axs = plt.subplots(n_obj, n_samples * 2, figsize=(2 * n_samples * 2, 2 * n_obj), dpi=74)
    for col in range(n_samples):
        for row in range(n_obj):
            ax = axs[row, col * 2]
            ax.axis('off')
            buf = x[row * n_samples + col].cpu().numpy()
            target = y[row * n_samples + col]
            ax.imshow(buf)

            ax = axs[row, col * 2 + 1]
            ax.axis('off')
            ax.text(0, 0, f'{target}')

    plt.show()

    targets = y.cpu().numpy()[:, 0]
    plt.hist(targets)
    plt.show()


def plot_data_loader_samples(n_obj, n_samples, data_loader):
    for X, y in data_loader:
        break

    _, axs = plt.subplots(n_obj, n_samples, figsize=(2 * n_samples, 2 * n_obj), dpi=74)
    for col in range(n_samples):
        for row in range(n_obj):
            ax = axs[row, col]
            ax.axis('off')
            ax.imshow(X[row * n_samples + col].cpu().numpy())
    plt.show()


def get_device(num_processes=1, device=None):
    if device is None:
        if num_processes <= 1:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = 'cpu'
        else:
            device = 'cpu'

    print(f'Device "{device}" selected')
    return torch.device(device)
