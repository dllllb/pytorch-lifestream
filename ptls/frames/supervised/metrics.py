import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional.classification import auroc

from ptls.loss import cross_entropy, kl, mape_metric, mse_loss, r_squared

logger = logging.getLogger(__name__)


class EpochAuroc(torchmetrics.Metric):
    """Deprecated. Use `torchmetrics.AUROC`
    """
    def __init__(self):
        super().__init__(compute_on_step=False)

        self.add_state('y_hat', default=[])
        self.add_state('y', default=[])

    def update(self, y_hat, y):
        self.y_hat.append(y_hat)
        self.y.append(y)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return auroc(y_hat, y.long())


class DistributionTargets(torchmetrics.Metric):
    def __init__(self, col_name):
        super().__init__(compute_on_step=False)

        self.add_state('y_hat', default=[])
        self.add_state('y', default=[])
        self.col_name = col_name
        self.sign = -1 if self.col_name == 'neg_sum' else 1

    def update(self, y_hat, y):
        y_hat = y_hat[self.col_name]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y = torch.tensor(np.array(y[self.col_name].tolist(), dtype='float64'), device=device)
        self.y_hat.append(y_hat)
        self.y.append(y)


class CrossEntropy(DistributionTargets):
    def __init__(self, col_name):
        super().__init__(col_name)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return cross_entropy(y_hat, y)


class KL(DistributionTargets):
    def __init__(self, col_name):
        super().__init__(col_name)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return kl(y_hat, y)


class MSE(DistributionTargets):
    def __init__(self, col_name):
        super().__init__(col_name)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return mse_loss(y_hat, torch.log(self.sign * y[:, None] + 1))


class MAPE(DistributionTargets):
    def __init__(self, col_name):
        super().__init__(col_name)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return mape_metric(self.sign * torch.exp(y_hat - 1), y[:, None])


class LogAccuracy(torchmetrics.Metric):
    def __init__(self, **params):
        super().__init__(**params)

        self.add_state('correct', default=torch.tensor(0.0))
        self.add_state('total', default=torch.tensor(0.0))

    def update(self, preds, target):
        if len(preds.shape) == 1:
            preds = (preds > 0.5).to(dtype=target.dtype)
        else:
            preds = torch.argmax(preds, dim=1)
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class R_squared(DistributionTargets):
    def __init__(self, col_name):
        super().__init__(col_name)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return r_squared(self.sign * torch.exp(y_hat - 1), y[:, None])


class RMSE(torchmetrics.Metric):
    """
    RMSE/MAE for multiple outputs with exponential scaler.
    Should be elaborated via adaptor-classes like in [ptls/nn/trx_encoder/scalers.py].
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, calc_mae=True):
        super().__init__()
        self.calc_mae = calc_mae
        self.add_state("psum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pnum", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y1, y):
        if y.dim() == 1:
            delta = y1 - y if y1.dim() == 1 else y1[:, 0].exp() - y
        else:
            delta = y1[:, 0].exp() - y.sum(dim=1)
        self.psum += torch.sum(delta.abs() if self.calc_mae else delta.square())
        self.pnum += y.shape[0]

    def compute(self):
        res = self.psum / self.pnum
        return res if self.calc_mae else res.sqrt()


class BucketAccuracy(torchmetrics.Metric):
    """
    Multiclass Accuracy for bucketized regression variable.
    Should be elaborated via adaptor-classes like in [ptls/nn/trx_encoder/scalers.py].
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, n_buckets=10, compute_on_cpu=True):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.n_buckets = n_buckets
        self.add_state("y1", default=[])
        self.add_state("y", default=[])

    def update(self, y1, y):
        self.y1.append(y1 if y1.dim() == 1 else y1[:, 0].exp())
        self.y.append(y if y.dim() == 1 else y.sum(dim=1))

    def compute(self):
        y1, y = torch.cat(self.y1).float(), torch.cat(self.y).float()
        q = torch.linspace(0, 1, self.n_buckets + 1, dtype=y.dtype, device=y.device)[1:]
        buckets = torch.quantile(y, q, interpolation="nearest")
        y1 = torch.bucketize(y1, buckets, out_int32=True)
        y = torch.bucketize(y, buckets, out_int32=True)
        return torchmetrics.functional.accuracy(y1, y)


class KLDiv(torchmetrics.Metric):
    """
    KL divergence for distribution-like target.
    Should be elaborated via adaptor-classes like in [ptls/nn/trx_encoder/scalers.py].
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("psum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pnum", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y1, y):
        prob, ysum = F.softmax(y1[:, 3:], dim=1), y.sum(dim=1)
        kl = torchmetrics.functional.kl_divergence(y, prob, reduction=None)
        self.psum += torch.sum(kl.where(ysum > 0, -F.logsigmoid(y1[:, 2])))
        self.pnum += y.shape[0]

    def compute(self):
        return self.psum / self.pnum
