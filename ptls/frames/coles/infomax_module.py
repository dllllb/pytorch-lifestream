from ptls.frames.abs_module import ABSModule
import torchmetrics
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from torch import nn as nn
import torch


class InfoMaxModule(ABSModule):
    def __init__(self,
                 seq_encoder: SeqEncoderContainer = None,
                 local_loss=None,
                 coles_loss=None,
                 sampler=None,
                 add_inner_neg_loss=True,
                 head=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):

        if validation_metric is None:
            validation_metric = torchmetrics.MeanMetric()

        super().__init__(validation_metric,
                         seq_encoder,
                         local_loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        self._head = head
        self.sampler = sampler
        self._coles_loss = coles_loss
        self.add_inner_neg_loss = add_inner_neg_loss

    @property
    def metric_name(self):
        return 'val_loss'
        
    def get_embs_forward(self, chains, slices):
        chains_vectors = self._seq_encoder(chains)
        slice_vectors = self._seq_encoder(slices)
        if self._head is not None:
            chains_vectors = self._head(chains_vectors)
            slice_vectors = self._head(slice_vectors)
        return chains_vectors, slice_vectors
    
    def validation_step(self, batch, _):
        encoded_chains, encoded_slices, ranges, pos_slice_count, inner_neg_splits, inner_neg_slices = self.shared_step(batch)
        if self._coles_loss is not None:
            _, _, neg_chains, neg_slices = self.sampler.get_pairs(encoded_chains[::pos_slice_count], encoded_slices, pos_slice_count)
            coles_loss = self._coles_loss(encoded_chains, encoded_slices, neg_chains, neg_slices)
            inner_negative_loss = nn.functional.relu(0.5 - torch.sqrt(torch.sum((inner_neg_splits - inner_neg_slices) ** 2, dim=-1))).pow(2).mean()
            loss = coles_loss
            if self.add_inner_neg_loss:
                loss = (loss + inner_negative_loss) / 2
            self._validation_metric(loss)
        
    def shared_step(self, batch):
        chains, slices, ranges, pos_slice_count, inner_pos_count, all_slices_count = batch
        encoded_chains, encoded_slices = self.get_embs_forward(chains, slices)
        inner_samples = encoded_slices[all_slices_count:]
        encoded_slices = encoded_slices[:all_slices_count]
        neg_size = len(inner_samples)
        inner_neg_splits = inner_samples[:neg_size//2]
        inner_neg_slices = inner_samples[neg_size//2:]
        return encoded_chains, encoded_slices, ranges, pos_slice_count, inner_neg_splits, inner_neg_slices
    
    def training_step(self, batch, batch_idx):
        encoded_chains, encoded_slices, ranges, pos_slice_count, inner_neg_splits, inner_neg_slices = self.shared_step(batch)
        loss = 0.0
        
        if self._coles_loss is not None:
            _, _, neg_chains, neg_slices = self.sampler.get_pairs(encoded_chains[::pos_slice_count], encoded_slices, pos_slice_count)
            loss += self._coles_loss(encoded_chains, encoded_slices, neg_chains, neg_slices)
            inner_negative_loss = nn.functional.relu(0.5 - torch.sqrt(torch.sum((inner_neg_splits - inner_neg_slices) ** 2, dim=-1))).pow(2).mean()
            if self.add_inner_neg_loss:
                loss = (loss + inner_negative_loss) / 2
            self.log("loss", loss)

        return loss
    
    def is_requires_reduced_sequence(self):
        return True
