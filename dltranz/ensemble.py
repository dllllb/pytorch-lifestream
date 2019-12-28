import torch
import numpy as np
from torch import nn


class ModelEnsemble(nn.Module):
    def __init__(self, submodels):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(submodels)

    def forward(self, *args):
        out = torch.cat([m(*args) for m in self.models])
        return out.mean(dim=0)


def pred_crossed(pred, thresh, shift=0):
    res = np.zeros((pred.shape[0],))
    centered = np.array(pred) - thresh
    for timestep in range(shift, pred.shape[1] - 1):
        cur_pred = centered[:, timestep]
        next_pred = centered[:, timestep + 1]

        res = res + np.logical_or(
            np.logical_and(cur_pred < 0, next_pred > 0),
            np.logical_and(cur_pred > 0, next_pred < 0))
    return res


class UncertainModelEnsemble(nn.Module):
    def __init__(self, submodels, uncertainty_thr=0):
        super().__init__()
        self.models = nn.ModuleList(submodels)
        self.uncertainty_thr = uncertainty_thr
        print([(m.max_val, m.min_val) for m in self.models])

    def forward(self, *args):
        out = torch.cat([m(*args) for m in self.models])
        out_pred = out[:, :, -1, 0].mean(dim=0)

        out_uncertain = out.mean(dim=0).squeeze()
        out_uncertain = pred_crossed(out_uncertain.payload.cpu(), self.uncertainty_thr)
        return out_pred, out_uncertain
