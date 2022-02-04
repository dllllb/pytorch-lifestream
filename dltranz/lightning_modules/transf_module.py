import torch.nn

from dltranz.lightning_modules.AbsModule import ABSModule
from dltranz.metric_learn.metric import BatchRecallTopPL
from dltranz.models import create_head_layers
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.trx_encoder import PaddedBatch


class TransformerModule(ABSModule):
    def __init__(self, params):
        super().__init__(params)

        self.trx_cat_encoder = self.seq_encoder.model[0]
        if type(self.trx_cat_encoder) is torch.nn.Sequential:
            self.trx_cat_encoder = self.trx_cat_encoder[0]
        self.transf_encoder = self.seq_encoder.model[1]

        trx_size = self.trx_cat_encoder.output_size
        hid_size = self.seq_encoder.embedding_size
        self.trx_encoder = torch.nn.Linear(trx_size, hid_size)
        self.trx_decoder = torch.nn.Linear(hid_size, trx_size)

        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, hid_size))

        self.loss_coles = self._loss

    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def get_loss(self):
        sampling_strategy = get_sampling_strategy(self.hparams.params)
        loss = get_loss(self.hparams.params, sampling_strategy)
        return loss

    def get_validation_metric(self):
        return BatchRecallTopPL(**self.hparams.params['validation_metric_params'])

    def shared_step(self, x, y):
        y_h = self(x)
        return y_h, y

    def forward(self, x):
        z_trx = self.trx_cat_encoder(x)
        z_trx_in = PaddedBatch(self.trx_encoder(z_trx.payload), z_trx.seq_lens)

        z_seq_out = self.transf_encoder(z_trx_in)
        z_cls = z_seq_out.payload[:, 0, :]
        z_seq = PaddedBatch(
            self.trx_decoder(z_seq_out.payload[:, 1:, :]),
            z_seq_out.seq_lens)
        return z_cls

    def training_step(self, batch, _):
        x, y = batch

        z_trx = self.trx_cat_encoder(x)
        z_trx_in = PaddedBatch(self.trx_encoder(z_trx.payload), z_trx.seq_lens)
        trx_mask = self.get_trx_mask(z_trx_in)
        z_trx_masked = self.mask_trx(z_trx_in, trx_mask)

        z_seq_out = self.transf_encoder(z_trx_masked)
        z_cls = z_seq_out.payload[:, 0, :]
        z_seq = PaddedBatch(
            self.trx_decoder(z_seq_out.payload[:, 1:, :]),
            z_seq_out.seq_lens)

        loss_coles = self.loss_coles(z_cls, y)
        self.log('loss/coles', loss_coles)

        loss_mlm, loss_var = self.calc_loss_mask(z_seq, z_trx, trx_mask)
        self.log('loss/mlm', loss_mlm)
        self.log('loss/var', loss_var)

        loss = sum([
            loss_coles,
            loss_mlm * self.hparams.params['train.mlm_loss.loss_mlm_w'],
            loss_var * self.hparams.params['train.mlm_loss.loss_var_w'],
        ])
        return loss

    def get_trx_mask(self, z_trx: PaddedBatch):
        B, T, H = z_trx.payload.size()
        device = z_trx.payload.device
        mask_len = (1 - torch.triu(torch.ones(T, T, device=device), 1))[z_trx.seq_lens.long() - 1]

        mask = torch.bernoulli(mask_len * self.hparams.params['train.mlm_loss.replace_prob']).bool()
        return mask

    def mask_trx(self, z_trx: PaddedBatch, mask):
        B, T, H = z_trx.payload.size()
        return PaddedBatch(
            torch.where(
                mask.unsqueeze(2).repeat(1, 1, H),
                self.token_mask.repeat(B, T, 1),
                z_trx.payload,
            ),
            z_trx.seq_lens,
        )

    def calc_loss_mask(self, z_seq: PaddedBatch, z_trx: PaddedBatch, trx_mask):
        B, T, H = z_trx.payload.size()
        var_gamma = self.hparams.params['train.mlm_loss.var_gamma']

        mask = trx_mask.view(B * T)
        pred_tensors = z_seq.payload.view(B * T, H)[mask]
        true_tensors = z_trx.payload.view(B * T, H)[mask]

        loss_mlm = (pred_tensors - true_tensors).pow(2).sum(dim=1).mean()
        loss_var = torch.relu(var_gamma - torch.var(pred_tensors, dim=0)).mean()

        return loss_mlm, loss_var
