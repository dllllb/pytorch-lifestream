import torch
from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import ContrastiveLoss, CLUBLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.data_load.padded_batch import PaddedBatch
from itertools import chain
from copy import deepcopy


class MultiCoLESModule(ABSModule):
    def __init__(self,
                 seq_encoder: SeqEncoderContainer = None,
                 discriminator=None,
                 head=None,
                 loss=None,
                 discriminator_loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 d_optimizer_partial=None,
                 lr_scheduler_partial=None,
                 trained_encoders=None,
                 coles_coef=1.,
                 embed_coef=1.,
                 g_step_every=1):

        assert discriminator is not None and d_optimizer_partial is not None
        #assert (seq_encoder.n_encoders == 1) != (trained_encoders is not None)
        self.coles_coef = coles_coef
        self.embed_coef = embed_coef
        self.g_step_every = g_step_every

        if head is None:
            head = Head(use_norm_encoder=True)

        if loss is None:
            loss = ContrastiveLoss(margin=1.,
                                   sampling_strategy=HardNegativePairSelector(neg_count=5))

        if validation_metric is None:
            validation_metric = BatchRecallTopK(K=4, metric='cosine')

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        if discriminator_loss is None:
            self.discriminator_loss = CLUBLoss(emb_coef=1., prob_coef=1.)
        else:
            self.discriminator_loss = discriminator_loss

        self.trained_models = None
        if trained_encoders is not None:
            self.trained_models = torch.nn.ModuleList([deepcopy(seq_encoder) for _ in trained_encoders])
            for i, enc_path in enumerate(trained_encoders):
                self.trained_models[i].load_state_dict(torch.load(enc_path))
                for param in self.trained_models[i].parameters():
                    param.requires_grad = False

        self.automatic_optimization = False
        self.discriminator = discriminator
        self.d_optimizer_partial = d_optimizer_partial

        self._head = head

    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def lr_scheduler_step(
            self,
            scheduler,
            optimizer_idx,
            metric,
    ) -> None:
        scheduler.step()

    def shared_step(self, x, y):
        if self.trained_models is not None:
            out_h = list()
            for m in self.trained_models:
                y_h_ = m(x)
                if self._head is not None:
                    y_h_ = self._head(y_h_)
                out_h.append(y_h_)
            domain_a = torch.cat(out_h, dim=-1)
            domain_b = self(x)
            return [domain_a, domain_b], y

        else:
            y_h = self(x)
            if self._head is not None:
                y_h = self._head(y_h)
            return y_h, y

    def training_step(self, batch, batch_idx):
        opt, d_opt = self.optimizers()
        (domain_a, domain_b), y = self.shared_step(*batch)
        coles_info, embed_info = dict(), dict()

        # d opt
        domain_b_pred = self.discriminator(domain_a.detach())
        d_loss, d_info = self.discriminator_loss.pred_loss(domain_b.detach(), domain_b_pred)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        # g opt
        if batch_idx % self.g_step_every == 0:
            coles_loss, coles_info = self._loss(domain_b, y)
            embed_loss, embed_info = self.discriminator_loss.embed_loss(domain_b, domain_b_pred)
            loss = self.coles_coef * coles_loss + self.embed_coef * embed_loss
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        for k, v in chain(d_info.items(), coles_info.items(), embed_info.items()):
            self.log(k, v)

        if type(batch) is tuple:
            x, y = batch
            if isinstance(x, PaddedBatch):
                self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
        else:
            # this code should not be reached
            self.log('seq_len', -1, prog_bar=True)
            raise AssertionError('batch is not a tuple')

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self._seq_encoder.parameters())
        d_optimizer = self.d_optimizer_partial(self.discriminator.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer, d_optimizer], [scheduler]
