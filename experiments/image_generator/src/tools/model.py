import pickle

import torch

from tools.shape_generator import ShapeGenerator


class Flatten(torch.nn.Module):
    def forward(self, t):
        B, C, W, H = t.size()
        return t.view(B, C * W * H)


class L2Normalization(torch.nn.Module):
    def forward(self, t):
        return t.div(torch.norm(t, dim=1).view(-1, 1))


class CnnModel(torch.nn.Module):
    vector_size = 64

    def __init__(self, image_shape):
        super().__init__()

        self.image_shape = image_shape

        dropout_p = 0.0  # 0.2

        layers = [
            # 148
            torch.nn.Conv2d(
                in_channels=self.image_shape[2],
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            # torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.MaxPool2d(kernel_size=2),
            # 74
            torch.nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            # torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.MaxPool2d(kernel_size=2),
            # 36
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            # torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.MaxPool2d(kernel_size=2),
            # 18
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
            ),
            # torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.MaxPool2d(kernel_size=2),
            # 8
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            # torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.MaxPool2d(kernel_size=2),
            # 4
            Flatten(),
            torch.nn.Linear(32 * 4 * 4, self.vector_size),
            # torch.nn.ReLU(),

            L2Normalization(),
        ]

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        t = x

        # B, H, W, C => B, C, H, W
        t = t.transpose(1, 3).transpose(2, 3)

        t = self.layers(t)

        return t


class MultiHead(torch.nn.Module):
    def __init__(self, heads):
        super().__init__()

        self.heads = torch.nn.ModuleDict(heads)

    def forward(self, t):
        result = {k: v(t) for k, v in self.heads.items()}
        return result


def test_cnn():
    model = CnnModel(ShapeGenerator.image_shape)
    out = model(torch.rand(5, 148, 148, 4))
    assert out.size() == (5, 64)


def freeze_layers(model):
    for p in model.parameters():
        p.requires_grad = False


def copy_model(model):
    return pickle.loads(pickle.dumps(model))
