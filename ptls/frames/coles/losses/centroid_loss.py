import torch
from torch import nn as nn


class CentroidLoss(nn.Module):
    """Centroid loss

    Class centroids are calculated over batch

    Parameters
    ----------
    class_num
        set the number of classes if you know limit (e.g. 2 for binary classifications)
        set None for metric learning task with unknown class number
    centroid_margin
        l2 distance between the class centers, closer than which the loss will be calculated.
        Class centers tend to be further than `centroid_margin`.
    """

    def __init__(self, class_num, centroid_margin=1.4):
        super().__init__()
        self.class_num = class_num
        if self.class_num is not None:
            self.register_buffer('l_targets_eye', torch.eye(class_num))
        self.centroid_margin = centroid_margin
        self.eps = 1e-6

    def forward(self, embeddings, target):
        if self.class_num is None:
            class_num = target.max() + 1
            l_targets_eye = torch.eye(class_num, device=target.device)
        else:
            l_targets_eye = self.l_targets_eye

        l_targets_ohe = l_targets_eye[target]  # B, class
        class_centers = embeddings.unsqueeze(1) * l_targets_ohe.unsqueeze(2) # B, class, H
        class_centers = class_centers.sum(dim=0).div(l_targets_ohe.sum(dim=0).unsqueeze(1) + self.eps)  # class, H

        cc_pairs = (class_centers.unsqueeze(1) - class_centers.unsqueeze(0)).pow(2).sum(dim=2)
        cc_pairs = torch.relu(self.centroid_margin - (cc_pairs + self.eps).pow(0.5)).pow(2)
        cc_pairs = torch.triu(cc_pairs, diagonal=1).sum() / cc_pairs.size(0) * (cc_pairs.size(0) - 1) / 2

        distances = embeddings - class_centers[target]
        l2_loss = distances.pow(2).sum(dim=1)  # B,

        return l2_loss.mean() + cc_pairs


class CentroidSoftmaxLoss(nn.Module):
    """Centroid Softmax loss

    Class centroids are calculated over batch

    Parameters
    ----------
    class_num
        set the number of classes if you know limit (e.g. 2 for binary classifications)
        set None for metric learning task with unknown class number
    temperature:
        temperature for softmax logits for scaling l2 distance on unit sphere
    """

    def __init__(self, class_num, temperature=10.0):
        super().__init__()
        self.class_num = class_num
        if self.class_num is not None:
            self.register_buffer('l_targets_eye', torch.eye(class_num))
        self.eps = 1e-6
        self.temperature = temperature

    def forward(self, embeddings, target):
        if self.class_num is None:
            class_num = target.max() + 1
            l_targets_eye = torch.eye(class_num, device=target.device)
        else:
            l_targets_eye = self.l_targets_eye

        l_targets_ohe = l_targets_eye[target]  # B, class
        class_centers = embeddings.unsqueeze(1) * l_targets_ohe.unsqueeze(2) # B, class, H
        class_centers = class_centers.sum(dim=0).div(l_targets_ohe.sum(dim=0).unsqueeze(1) + self.eps)  # class, H

        distances = embeddings.unsqueeze(1) - class_centers.unsqueeze(0)  # B, class, H
        l2_loss = distances.pow(2).sum(dim=2)  # B, class

        l2_loss = torch.softmax(-l2_loss * self.temperature, dim=1)
        l2_loss = l2_loss[torch.arange(l2_loss.size(0), device=embeddings.device), target]
        l2_loss = -torch.log(l2_loss)

        return l2_loss.mean()
