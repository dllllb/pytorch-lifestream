import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Dataset

from tools.shape_generator import ShapeGenerator


class ImageTransformer(Dataset):
    def __init__(self, data, transformer):
        super().__init__()

        self.data = data
        self.transformer = transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        obj = self.data[item]
        if type(obj) is tuple:
            x, y = obj
            if self.transformer is not None:
                x = self.transformer(x)
            return x, y
        else:
            if self.transformer is not None:
                obj = self.transformer(obj)
            return obj


class CombineTransformer:
    def __init__(self, t_list):
        self.t_list = t_list

    def __call__(self, x):
        y = x
        for f in self.t_list:
            y = f(y)
        return y


class AddTransformer:
    def __init__(self, transformer, alpha):
        self.transformer = transformer
        self.alpha = alpha

    def __call__(self, x):
        y = self.transformer(x) * self.alpha + x * (1.0 - self.alpha)
        return y


class FillColorTransformer:
    def __init__(self, color):
        self.color = color

    def __call__(self, x):
        y = np.ones_like(x)
        y[:, :, 0:3] = self.color
        return y


class RandomColorTransformer:
    def __call__(self, x):
        y = np.ones_like(x)
        y[10:-10, 10:-10, 0:3] = np.random.rand(3)
        y[30:-30, 30:-30, 0:3] = np.random.rand(3)
        return y


class RandomTransformer:
    def __call__(self, x):
        y = np.random.rand(np.prod(x.shape)).reshape(x.shape).astype(x.dtype)
        y[:, :, 3] = 1.0
        return y


class BlurTransformer:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        from scipy.ndimage.filters import gaussian_filter

        y = np.ones_like(x)
        y[:, :, 0:3] = gaussian_filter(x[:, :, 0:3], sigma=self.sigma, order=0)
        return y


class ChannelBlurTransformer:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        from scipy.ndimage.filters import gaussian_filter

        y = np.ones_like(x)
        for c in range(3):
            y[:, :, c] = gaussian_filter(x[:, :, c], sigma=self.sigma, order=0)
        return y


def test_transformers():
    test_fn = [
        ('FillColorTransformer(None)', FillColorTransformer(None)),
        ('FillColorTransformer(red)', FillColorTransformer(np.array([1.0, 0.0, 0.0]))),
        ('FillColorTransformer(gray)', FillColorTransformer(np.array([0.5, 0.5, 0.5]))),
        ('RandomTransformer', RandomTransformer()),
        ('BlurTransformer 0.5', BlurTransformer(0.5)),
        ('BlurTransformer 1.5', BlurTransformer(1.5)),
        ('ChannelBlurTransformer 0.5', ChannelBlurTransformer(0.5)),
        ('ChannelBlurTransformer 1.5', ChannelBlurTransformer(1.5)),
        ('ChannelBlurTransformer 5.0', ChannelBlurTransformer(5.0)),
        ('RandomBlur 0.4', CombineTransformer([RandomTransformer(), BlurTransformer(0.4)])),
        ('RandomBlur 1.0', CombineTransformer([RandomTransformer(), BlurTransformer(1.0)])),
        ('RandomBlur 1.5', CombineTransformer([RandomTransformer(), BlurTransformer(1.5)])),
        ('RandomChannelBlurTransformer 0.4', CombineTransformer([RandomTransformer(), ChannelBlurTransformer(0.4)])),
        ('RandomChannelBlurTransformer 1.0', CombineTransformer([RandomTransformer(), ChannelBlurTransformer(1.0)])),
        ('RandomChannelBlurTransformer 1.5', CombineTransformer([RandomTransformer(), ChannelBlurTransformer(1.5)])),
    ]

    sg = ShapeGenerator(keep_params=[])
    _, axs = plt.subplots(len(test_fn), 2, figsize=(4, 2 * len(test_fn)), dpi=74)

    for row, (name, f) in enumerate(test_fn):
        x = sg.get_buffer()[0]
        y = f(x)

        ax = axs[row][0]
        ax.axis('off')
        ax.imshow(x)
        ax.set_title('Original')

        ax = axs[row][1]
        ax.axis('off')
        ax.imshow(y)
        ax.set_title(name)

        assert x.shape == y.shape

    plt.show()
