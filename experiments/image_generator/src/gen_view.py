import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tools import logging_base_config, get_device
from generator import Generator, z_size, plot_generated

generator_name = '../models/generator-crop.p.0128'


def plot_var(n_obj, n_samples, generator, title, viriation=1.0):
    figure, axs = plt.subplots(n_obj, n_samples, figsize=(2 * n_samples, 2 * n_obj), dpi=74)
    z0 = torch.normal(
        torch.zeros((1, z_size)),
        torch.ones((1, z_size)),
    ).to(device=device)
    for row in range(n_obj):
        # z = torch.zeros((n_samples, z_size)).to(device=device)
        z = z0.repeat(n_samples, 1)

        z[:, row] += torch.arange(-viriation, viriation + 1e-6, viriation * 2.0 / (n_samples - 1))

        data = generator(z).detach().cpu().numpy()
        # data = np.clip(data / 2.0 + 0.5, 0.0, 1.0)
        data = np.clip((data - 0.1) / 0.8, 0.0, 1.0)

        for col in range(n_samples):
            ax = axs[row, col]
            ax.axis('off')
            ax.imshow(data[col])
    figure.suptitle(title)
    plt.show()


if __name__ == '__main__':
    logging_base_config()
    device = get_device(device='cpu')

    generator = torch.load(generator_name)
    generator.to(device=device)
    generator.eval()
    plot_var(z_size, 7, generator, f'[{generator_name}] variates', viriation=1.0)

    plot_generated(8, 8, generator, f'[{generator_name}] samples', device)
