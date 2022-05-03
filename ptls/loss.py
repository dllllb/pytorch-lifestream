import torch
from torch import nn
import numpy as np

from ptls.trx_encoder import PaddedBatch


def cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    device = pred.device
    return torch.mean(torch.sum(-soft_targets.to(device) * logsoftmax(pred), 1))


def kl(pred, soft_targets):
    eps = 1e-7
    softmax = torch.nn.Softmax(dim=1)
    device = pred.device
    return torch.mean(torch.sum(soft_targets.to(device) * torch.log(soft_targets.to(device) / (softmax(pred) + eps) + eps), 1))


def mse_loss(pred, actual):
    device = pred.device
    return torch.mean((pred - actual.to(device))**2)


def mape_metric(pred, actual):
    eps = 1
    device = pred.device
    return torch.mean((actual.to(device) - pred).abs() / (actual.to(device).abs() + eps))

def r_squared(pred, actual):
    device = pred.device
    return 1 - torch.sum((actual.to(device) - pred)**2) \
                         / torch.sum((actual.to(device) - torch.mean(actual.to(device)))**2)


class PairwiseMarginRankingLoss(nn.Module):
    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean'):
        """
        Pairwise Margin Ranking Loss. All setted parameters redirected to nn.MarginRankingLoss.
        All the difference is that pairs automatically generated for margin ranking loss.
        All possible pairs of different class are generated.
        """
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin, size_average, reduce, reduction)

    def forward(self, prediction, label):
        """
        Get pairwise margin ranking loss.
        :param prediction: tensor of shape Bx1 of predicted probabilities
        :param label: tensor of shape Bx1 of true labels for pair generation
        """

        # positive-negative selectors
        mask_0 = label == 0
        mask_1 = label == 1

        # selected predictions
        pred_0 = torch.masked_select(prediction, mask_0)
        pred_1 = torch.masked_select(prediction, mask_1)
        pred_1_n = pred_1.size()[0]
        pred_0_n = pred_0.size()[0]

        if pred_1_n > 0 and pred_0_n:
            # create pairs
            pred_00 = pred_0.unsqueeze(0).repeat(1, pred_1_n)
            pred_11 = pred_1.unsqueeze(1).repeat(1, pred_0_n).view(pred_00.size())
            out01 = -1 * torch.ones(pred_1_n*pred_0_n).to(prediction.device)

            return self.margin_loss(pred_00.view(-1), pred_11.view(-1), out01)
        else:
            return torch.sum(prediction) * 0.0


class MultiLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, pred, true):
        loss = 0
        for weight, criterion in self.losses:
            loss = weight * criterion(pred, true) + loss

        return loss

class TransactionSumLoss(nn.Module):
    def __init__(self, n_variables_to_predict):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.n_variables_to_predict = n_variables_to_predict

    def forward(self, pred, true):
        #the 0th element is not used, because it is reserved for the total sum of transactions
        #which is not predicted now
        loss = self.bce_with_logits(pred[:,1:self.n_variables_to_predict], true[:,1:self.n_variables_to_predict])
        return loss


class AllStateLoss(nn.Module):
    def __init__(self, point_loss):
        super().__init__()
        self.loss = point_loss

    def forward(self, pred: PaddedBatch, true):
        y = torch.cat([torch.Tensor([yb] * length) for yb, length in zip(true, pred.seq_lens)])
        weights = torch.cat([torch.arange(1, length + 1) / length for length in pred.seq_lens])

        loss = self.loss(pred, y, weights)

        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, pred, true):
        return self.loss(pred.float(), true.float())


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, true):
        return self.loss(pred.float(), true.float())


class PseudoLabeledLoss(nn.Module):
    def __init__(self, loss, pl_threshold=0.5, unlabeled_weight=1.):
        super().__init__()
        self.loss = loss
        self.pl_threshold = pl_threshold
        self.unlabeled_weight = unlabeled_weight

    def forward(self, pred, true):
        label_pred, unlabel_pred = pred['labeled'], pred['unlabeled']
        if isinstance(self.loss, nn.NLLLoss):
            pseudo_labels = torch.argmax(unlabel_pred.detach(), 1)
        elif isinstance(self.loss, BCELoss):
            pseudo_labels = (unlabel_pred.detach() > 0.5).type(torch.int64)
        else:
            raise Exception(f'unknown loss type: {self.loss}')

        # mask pseudo_labels, with confidence > pl_threshold
        if isinstance(self.loss, nn.NLLLoss):
            probs = torch.exp(unlabel_pred.detach())
            mask = (probs.max(1)[0] > self.pl_threshold)
        elif isinstance(self.loss, BCELoss):
            probs = unlabel_pred.detach()
            mask = abs(probs - (1 - pseudo_labels)) > self.pl_threshold
        else:
            mask = torch.ones(unlabel_pred.shape[0]).bool()

        Lloss = self.loss(label_pred, true)
        if mask.sum()==0:
            return Lloss
        else:
            Uloss = self.unlabeled_weight * self.loss(unlabel_pred[mask], pseudo_labels[mask])
            return (Lloss + Uloss) / (1 + self.unlabeled_weight)


class UnsupervisedTabNetLoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, output, obf_vars):
        embedded_x, y_pred = output
        errors = y_pred - embedded_x
        reconstruction_errors = torch.mul(errors, obf_vars) ** 2
        batch_stds = torch.std(embedded_x, dim=0) ** 2 + self.eps
        features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
        # compute the number of obfuscated variables to reconstruct
        nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
        # take the mean of the reconstructed variable errors
        features_loss = features_loss / (nb_reconstructed_variables + self.eps)
        # the mean per batch
        loss = torch.mean(features_loss)
        return loss


class DistributionTargetsLoss(nn.Module):
    def __init__(self, mult1=3, mult2=0.167, mult3=1, mult4=1):
        super().__init__()
        self.mults = [mult1, mult2, mult3, mult4]

    def forward(self, pred, true):
        log_sum_truth_neg = np.log(np.abs(true['neg_sum'].astype(np.float)) + 1)[:, None] if isinstance(true['neg_sum'], np.ndarray) else 0
        distribution_truth_neg = np.array(true['neg_distribution'].tolist()) if isinstance(true['neg_sum'], np.ndarray) else 0
        log_sum_truth_pos = np.log(np.abs(true['pos_sum'].astype(np.float)) + 1)[:, None]
        distribution_truth_pos = np.array(true['pos_distribution'].tolist())

        log_sum_hat_neg, distribution_hat_neg = pred['neg_sum'], pred['neg_distribution']
        log_sum_hat_pos, distribution_hat_pos = pred['pos_sum'], pred['pos_distribution']

        device = log_sum_hat_pos.device
        loss_sum_pos = mse_loss(log_sum_hat_pos, torch.tensor(log_sum_truth_pos, device=device).float())
        loss_distr_pos = cross_entropy(distribution_hat_pos, torch.tensor(distribution_truth_pos, device=device))
        if isinstance(true['neg_sum'], np.ndarray):
            loss_sum_neg = mse_loss(log_sum_hat_neg, torch.tensor(log_sum_truth_neg, device=device).float())
            loss_distr_neg = cross_entropy(distribution_hat_neg, torch.tensor(distribution_truth_neg, device=device))

        loss = loss_sum_neg * self.mults[0] + loss_sum_pos * self.mults[1] + loss_distr_neg * self.mults[2] + loss_distr_pos * self.mults[3] \
                    if isinstance(true['neg_sum'], np.ndarray) else loss_sum_pos * self.mults[1] + loss_distr_pos * self.mults[3]
        return loss.float()


def get_loss(params):
    loss_type = params['train.loss']

    if loss_type == 'bce':
        loss = BCELoss()
    elif loss_type == 'NLLLoss':
        loss = nn.NLLLoss()
    elif loss_type == 'ranking':
        loss = PairwiseMarginRankingLoss(margin=params['ranking.loss_margin'])
    elif loss_type == 'both':
        loss = MultiLoss([(1, BCELoss()), (1, PairwiseMarginRankingLoss(margin=params['ranking.loss_margin']))])
    elif loss_type == 'mae':
        loss = nn.L1Loss()
    elif loss_type == 'mse':
        loss = MSELoss()
    elif loss_type == 'transaction_sum':
        n_variables_to_predict = params['variable_predicted']
        loss = TransactionSumLoss(n_variables_to_predict)
    elif loss_type == 'unsupervised_tabnet':
        loss = UnsupervisedTabNetLoss()
    elif loss_type == 'pseudo_labeled':
        loss = PseudoLabeledLoss(
            loss=get_loss(params['labeled']),
            pl_threshold=params['pl_threshold'],
            unlabeled_weight=params['unlabeled_weight']
        )
    elif loss_type == 'distribution_targets':
        loss = DistributionTargetsLoss()
    else:
        raise Exception(f'unknown loss type: {loss_type}')

    if params.get('head', {}).get('pred_all_states_loss', False):
        loss = AllStateLoss(loss)

    return loss

