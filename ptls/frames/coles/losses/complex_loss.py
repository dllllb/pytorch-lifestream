import torch


class ComplexLoss(torch.nn.Module):
    """Works like `ptls.loss.MultiLoss`

    """
    def __init__(self, ml_loss, aug_loss, ml_loss_weight=1.):
        super(ComplexLoss, self).__init__()
        self.aug_loss = aug_loss
        self.ml_loss = ml_loss
        self.ml_loss_weight = ml_loss_weight

    def forward(self, model_outputs, target):
        aug_output, ml_output = model_outputs
        aug_target = target[:, 0]
        ml_target = target[:, 1]
        aug = self.aug_loss(aug_output, aug_target) * (1 - self.ml_loss_weight)
        ml = self.ml_loss(ml_output, ml_target) * self.ml_loss_weight
        return aug + ml
