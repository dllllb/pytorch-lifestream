import torch
from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import MultiContrastiveLoss, CLUBLoss
from ptls.frames.coles.metric import MultiBatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder.utils import reset_parameters
from itertools import chain
from copy import deepcopy
from functools import partial


class ParallelModels(torch.nn.Module):
    def __init__(self, model_costructor, head_costructor=None, n_models=2):
        super().__init__()
        assert n_models == 2
        if head_costructor is None:
            head_costructor = partial(Head, use_norm_encoder=True)
        self.models = torch.nn.ModuleList([model_costructor() for _ in range(n_models)])
        self.heads = torch.nn.ModuleList([head_costructor() for _ in range(n_models)])

    def forward(self, inp, use_head=False):
        out = self.get_indep_preds(inp, use_head=use_head)
        return torch.cat(out, dim=-1)

    def get_indep_preds(self, inp, use_head=True):
        out = list()
        for m, h in zip(self.models, self.heads):
            x = m(inp)
            if use_head:
                x = h(x)
            out.append(x)
        return out

    @property
    def embedding_size(self):
        return sum([model.embedding_size for model in self.models])


class MultiCoLESSMLModule(ABSModule):
    def __init__(self,
                 seq_encoder_constructor,
                 head_constructor=None,
                 n_models=2,
                 discriminator=None,
                 loss=None,
                 discriminator_loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 d_optimizer_partial=None,
                 lr_scheduler_partial=None,
                 coles_coef=1.,
                 embed_coef=0.1):

        assert discriminator is not None and d_optimizer_partial is not None
        self.coles_coef = coles_coef
        self.embed_coef = embed_coef
        self.total_step = 0

        if loss is None:
            loss = MultiContrastiveLoss(margin=1.,
                                        sampling_strategy=HardNegativePairSelector(neg_count=5))

        if validation_metric is None:
            validation_metric = MultiBatchRecallTopK(n=n_models, K=4, metric='cosine')

        seq_encoder = ParallelModels(seq_encoder_constructor, head_constructor, n_models)

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        if discriminator_loss is None:
            self.discriminator_loss = CLUBLoss()
        else:
            self.discriminator_loss = discriminator_loss

        self.automatic_optimization = False
        self.discriminator = discriminator
        self.reference_discriminator = deepcopy(discriminator)
        reset_parameters(self.discriminator)
        reset_parameters(self.reference_discriminator)
        self.d_optimizer_partial = d_optimizer_partial

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
        y_h = self.seq_encoder.get_indep_preds(x, use_head=True)
        return y_h, y

    def training_step(self, batch, batch_idx):
        self.total_step += 1
        opt, d_opt = self.optimizers()
        (view_a, view_b), y = self.shared_step(*batch)

        # d opt
        random_inds = torch.randperm(view_b.shape[0])
        pos_preds = self.discriminator(view_a.detach(), view_b.detach())
        neg_preds = self.discriminator(view_a.detach(), view_b.detach()[random_inds])
        d_loss, d_info = self.discriminator_loss.pred_loss_prob(pos_preds, neg_preds)

        ref_pos_preds = self.reference_discriminator(view_a.detach(), view_b.detach())
        ref_neg_preds = self.reference_discriminator(view_a.detach(), view_b.detach()[random_inds])
        ref_d_loss, ref_d_info = self.discriminator_loss.pred_loss_prob(ref_pos_preds, ref_neg_preds)
        loss = d_loss + ref_d_loss

        d_opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(d_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        d_opt.step()

        # g opt
        coles_loss, coles_info = self._loss((view_a, view_b), y)

        pos_preds = self.discriminator(view_a, view_b)
        neg_preds = self.discriminator(view_a, view_b[random_inds])
        embed_loss, embed_info = self.discriminator_loss.embed_loss_prob(pos_preds, neg_preds)

        with torch.no_grad():
            ref_pos_preds = self.reference_discriminator(view_a.detach(), view_b.detach())
            ref_neg_preds = self.reference_discriminator(view_a.detach(), view_b.detach()[random_inds])
            ref_embed_loss, ref_embed_info = self.discriminator_loss.embed_loss_prob(ref_pos_preds, ref_neg_preds)

        loss = self.coles_coef * coles_loss + self.embed_coef * embed_loss
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()

        for k, v in chain(ref_d_info.items(), ref_embed_info.items()):
            self.log("ref_" + k, v)
        for k, v in chain(d_info.items(), coles_info.items(), embed_info.items()):
            self.log(k, v)
        self.log("embed_coef", self.embed_coef)

        if type(batch) is tuple:
            x, y = batch
            if isinstance(x, PaddedBatch):
                self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
        else:
            # this code should not be reached
            self.log('seq_len', -1, prog_bar=True)
            raise AssertionError('batch is not a tuple')

    def validation_step(self, batch, _):
        (view_a, view_b), y = self.shared_step(*batch)
        random_inds = torch.randperm(view_b.shape[0])
        pos_preds = self.discriminator(view_a.detach(), view_b.detach())
        neg_preds = self.discriminator(view_a.detach(), view_b.detach()[random_inds])
        ref_pos_preds = self.reference_discriminator(view_a.detach(), view_b.detach())
        ref_neg_preds = self.reference_discriminator(view_a.detach(), view_b.detach()[random_inds])
        coles_loss, coles_info = self._loss((view_a, view_b), y)
        d_loss, d_info = self.discriminator_loss.pred_loss_prob(pos_preds, neg_preds)
        embed_loss, embed_info = self.discriminator_loss.embed_loss_prob(pos_preds, neg_preds)
        ref_d_loss, ref_d_info = self.discriminator_loss.pred_loss_prob(ref_pos_preds, ref_neg_preds)
        ref_embed_loss, ref_embed_info = self.discriminator_loss.embed_loss_prob(ref_pos_preds, ref_neg_preds)
        for k, v in chain(ref_d_info.items(), ref_embed_info.items()):
            self.log("valid_ref_" + k, v)
        for k, v in chain(d_info.items(), coles_info.items(), embed_info.items()):
            self.log("valid_" + k, v)
        self._validation_metric((view_a, view_b), y)

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self._seq_encoder.parameters())
        d_optimizer = self.d_optimizer_partial(chain(self.discriminator.parameters(),
                                                     self.reference_discriminator.parameters()))
        scheduler = self._lr_scheduler_partial(optimizer)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer, d_optimizer], [scheduler]
