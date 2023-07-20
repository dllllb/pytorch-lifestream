import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Ids_to_idx():
    def __init__(self):
        self.label_dict = dict()
        self.cnt = 0
    def __call__(self, labels, device):
        idxs = []
        for l in labels:
            idx = self.label_dict.get(l, None)
            if idx is not None:
                idxs.append(idx)
            else:
                self.label_dict[l] = self.cnt
                idxs.append(self.cnt)
                self.cnt += 1
        return torch.LongTensor(idxs).to(device)

class ArcfaceLoss(nn.Module):
    r"""Implementation of large margin arc distance:
        from https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=10.0, m=0.2, easy_margin=False):
        super(ArcfaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.ids_to_idx = Ids_to_idx()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input, label):
        with open('debug.txt', 'a') as f:
            f.write(str(label))
            label = self.ids_to_idx(label, device = input.device)
            f.write(str(label) + '\n\n\n')
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        # print(output.shape)

        return self.criterion(output, label)

