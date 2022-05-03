from pyhocon import ConfigFactory
from ptls.seq_to_target import SequenceToTarget


class SeqToTargetDemo(SequenceToTarget):
    """pl.LightningModule for training CoLES embeddings
    Parameters
    ----------
    seq_encoder : torch.nn.Module
        sequence encoder
    encoder_lr : float. Default: 1e-4
        Learning rate for sequence encoder.
    in_features: int. Default: 256
        Dimension of embeddings from sequence encoder.
    out_features: int. Default: 4
        Dimension of output of this model (number of classes).
    head_lr: float. Default: 0.002
    	Learning rate for head of this model.
    weight_decay: float. Default: 0.0
        Weight decay for optimizer.
    lr_scheduler_step_size : int. Default: 5
        Period of learning rate decay.
    lr_scheduler_step_gamma: float. Default: 0.4
        Multiplicative factor of learning rate decay.
    """

    def __init__(self,
                 seq_encoder = None,
                 encoder_lr: float = 0.0001,
                 in_features: int = 256,
                 out_features: int = 4,
                 head_lr: float = 0.002,
                 weight_decay: float = 0.0,
                 lr_scheduler_step_size: int = 5,
		 lr_scheduler_step_gamma: float = 0.4):

        params = {
            'score_metric': ['accuracy'],

            'encoder_type': 'pretrained',
            'pretrained': {
                'pl_module_class': 'dltranz.lightning_modules.coles_module.CoLESModule',
                'lr': encoder_lr
            },

            'head_layers': [
              ['BatchNorm1d', {'num_features': in_features}],
              ['Linear', {"in_features": in_features, "out_features": out_features}],
              ['LogSoftmax', {'dim': 1}]
            ],

            'train': {
              'random_neg': 'false',
              'loss': 'NLLLoss',
              'lr': head_lr,
              'weight_decay': weight_decay,
            },
            'lr_scheduler': {
              'step_size': lr_scheduler_step_size,
              'step_gamma': lr_scheduler_step_gamma
            }
        }
        super().__init__(ConfigFactory.from_dict(params), seq_encoder)

