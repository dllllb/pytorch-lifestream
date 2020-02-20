"""
Генератор и дискриминатор из примера `dcgan.py` учатся, делают неплохие картинки:
- четко виден контур фона
- четко различаются два цвета
- шумов, зернистости и артефактов практически нет
- видны разные размеры полученных фигур
- расположение цетов чаще правильное (пятно в центре фона), иногда пятен несколько или они сползают на край
- у фигур нет явной формы, нет прямых линий
- часть фигур можно с нятяжкой причислить к правильным, они имеют выпуклый контур
- дополнительный слой апсемплинга, дает более выраженные грани на фигурах.
    Дискриминатор дает качество порядка 0.4 (было 0.01)
- еще один слой апсемплинга (4й), Как будто больше качества не добавляет.
    Дискриминатор не дает стабильного качества. Встречаются варианты [0.82, 0.28, 0.13, 0.25, 0.47, 0.44, ...]

- дискрминатор заменен на CNNModel
- линии и углы на фигурах стали более четкие

Такое качество достигается к 256 эпохе.
Дальше дискриминатор начинает побеждать, а генератор больше не учится.
"""

import logging

import torch
from torch.nn import Sequential
from torch.utils.data.dataloader import DataLoader

from tools import logging_base_config, get_device, plot_data_loader_samples

import numpy as np
import matplotlib.pyplot as plt
from tools.data_loader import ReplaceYDataset, FlattenMLDataset
from tools.dataset import ImageDataset, load_dataset, ImageMetricLearningDataset
import torch.nn as nn

from tools.dataset_transformers import ImageTransformer, CombineTransformer, AddTransformer, RandomTransformer, \
    ChannelBlurTransformer
from tools.model import CnnModel, freeze_layers
from tools.shape_generator import ShapeGenerator

logger = logging.getLogger(__name__)

num_processes = 1

z_size = 16

params = {
    # 'train_gen1_path': '../datasets/supervised-train_gen1-2000.p',
    # 'train_gen1_path': '../datasets/supervised-train_gen_blue_on_white-2000.p',
    'train_gen1_path': '../datasets/metric_learning-4000-4-color.p',

    'num_workers': 1,
    'train_batch_size': 64 // num_processes,

    'lr': 0.0002,

    'model_path_save_to': f'../models/generator-crop.p',
}


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.l1 = nn.Sequential(nn.Linear(z_size, 256 * (148 // 16) ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.Upsample(size=(148 // 8, 148 // 8)),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(size=(148 // 4, 148 // 4)),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(size=(148 // 2, 148 // 2)),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(size=(148, 148)),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 4, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, 148 // 16, 148 // 16)
        img = self.conv_blocks(out)

        # B, C, H, W => B, H, W, C
        img = img.transpose(2, 3).transpose(1, 3)

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(4, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 10
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        # B, H, W, C => B, C, H, W
        img = img.transpose(1, 3).transpose(2, 3)

        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def plot_generated(n_obj, n_samples, generator, title, device):
    z = torch.normal(
        torch.zeros((n_obj * n_samples, z_size)),
        torch.ones((n_obj * n_samples, z_size)),
    ).to(device=device)

    data = generator(z).detach().cpu().numpy()

    # data = np.clip(data + 0.5, 0.0, 1.0)
    data = np.clip(data, 0.0, 1.0)
    data = np.clip((data - 0.1) / 0.8, 0.0, 1.0)

    figure, axs = plt.subplots(n_obj, n_samples, figsize=(2 * n_samples, 2 * n_obj), dpi=74)
    for col in range(n_samples):
        for row in range(n_obj):
            ax = axs[row, col]
            ax.axis('off')
            ax.imshow(data[row * n_samples + col])
    figure.suptitle(title)
    plt.show()


if __name__ == '__main__':
    logging_base_config()
    device = get_device()

    ml_model = torch.load('../models/ml-4000-5-except_color-128-128.p')
    freeze_layers(ml_model)

    generator = Generator()
    discriminator = Sequential(
        ml_model,
        torch.nn.Linear(CnnModel.vector_size, 2048),
        torch.nn.Dropout(0.2),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1),
        torch.nn.Sigmoid(),
    )
    # discriminator = Discriminator()

    # train_data = ImageTransformer(
    #     data=FlattenMLDataset(ImageMetricLearningDataset(load_dataset(params['train_gen1_path']), 4),
    #                           torch.zeros(1, dtype=torch.float32)),
    #     transformer=AddTransformer(
    #         transformer=CombineTransformer([RandomTransformer(), ChannelBlurTransformer(2.0)]),
    #         alpha=0.1,
    #     ))
    train_data = FlattenMLDataset(ImageMetricLearningDataset(load_dataset(params['train_gen1_path']), 4),
                                  torch.zeros(1, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=params['train_batch_size'],
                              shuffle=True, num_workers=params['num_workers'])

    plot_data_loader_samples(4, 4, train_loader)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=params['lr'], betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=params['lr'], betas=(0.5, 0.999))

    loss_fn = torch.nn.BCELoss()

    generator.to(device)
    discriminator.to(device)
    generator.share_memory()
    discriminator.share_memory()

    n_epoch = 512 + 256
    d_skip = ''

    total_batch = 0
    c_loss_g = 0.0
    c_loss_d = 0.0
    alpha = 0.95
    for epoch in range(1, n_epoch + 1):
        for i, (real_data, real_target) in enumerate(train_loader):
            real_data = real_data.to(device=device)
            real_target = real_target.to(device=device)

            batch_size = real_data.size()[0]

            # real_data = (real_data - 0.5)
            real_data = real_data * 0.8 + 0.1

            z = torch.normal(
                torch.zeros((batch_size, z_size)),
                torch.ones((batch_size, z_size)),
            ).to(device=device)
            fake_data = generator(z)
            fake_target = torch.ones((batch_size, 1), dtype=real_target.dtype, device=device)

            #  Train Generator
            optimizer_g.zero_grad()
            g_loss = loss_fn(discriminator(fake_data), real_target)
            g_loss.backward()
            optimizer_g.step()

            # Train Discriminator
            optimizer_d.zero_grad()
            real_loss = loss_fn(discriminator(real_data), real_target)
            fake_loss = loss_fn(discriminator(fake_data.detach()), fake_target)
            d_loss = (real_loss + fake_loss) / 2.0
            d_loss.backward()
            optimizer_d.step()

            c_loss_g = alpha * c_loss_g + (1.0 - alpha) * g_loss.item()
            c_loss_d = alpha * c_loss_d + (1.0 - alpha) * d_loss.item()

            if total_batch % 50 == 0:
                print(f"[Epoch {epoch:4d}/{n_epoch}] [Batch {i:4d}/{len(train_loader)}] "
                      f"[D loss: {d_loss.item():.5f}] [G loss: {g_loss.item():.5f}] "
                      f"[CD loss: {c_loss_d:.5f}] [CG loss: {c_loss_g:.5f}] "
                      )
            if total_batch % 1000 == 0:
                plot_generated(4, 4, generator, title=f'epoch: {epoch}, total_batch: {total_batch}', device=device)
            total_batch += 1

            if epoch % 64 == 0:
                torch.save(generator, params['model_path_save_to'] + f'.{epoch:04d}')
