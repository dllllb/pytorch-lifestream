import torch
from torch import nn as nn
from torch.nn import functional as F


class CPC_Loss(nn.Module):
    def __init__(self, n_negatives=None, n_forward_steps=None):
        super().__init__()
        self.n_negatives = n_negatives
        self.n_forward_steps = n_forward_steps

    def _get_preds(self, base_embeddings, mapped_ctx_embeddings):
        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        _, _, _, n_forward_steps = mapped_ctx_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        device = mapped_ctx_embeddings.payload.device

        len_mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
        len_mask = (len_mask < seq_lens.unsqueeze(1).expand(-1, max_seq_len)).float()

        possible_negatives = base_embeddings.payload.view(batch_size * max_seq_len, emb_size)

        mask = len_mask.unsqueeze(0).expand(batch_size, *len_mask.shape).clone()

        mask = mask.reshape(batch_size, -1)
        sample_ids = torch.multinomial(mask, self.n_negatives)
        neg_samples = possible_negatives[sample_ids]

        positive_preds, neg_preds = [], []
        len_mask_exp = len_mask.unsqueeze(-1).unsqueeze(-1).to(device).expand(-1, -1, emb_size, n_forward_steps)
        trimmed_mce = mapped_ctx_embeddings.payload.mul(len_mask_exp)  # zero context vectors by sequence lengths
        for i in range(1, n_forward_steps + 1):
            ce_i = trimmed_mce[:, 0:max_seq_len - i, :, i - 1]
            be_i = base_embeddings.payload[:, i:max_seq_len]

            positive_pred_i = ce_i.mul(be_i).sum(axis=-1)
            positive_preds.append(positive_pred_i)

            neg_pred_i = ce_i.matmul(neg_samples.transpose(-2, -1))
            neg_preds.append(neg_pred_i)

        return positive_preds, neg_preds

    def forward(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        device = mapped_ctx_embeddings.payload.device
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)
        batch_size, max_len, emb_size = base_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        step_losses = []
        for i, (positive_pred_i, neg_pred_i) in enumerate(zip(positive_preds, neg_preds)):
            positions = torch.arange(max_len-i-1, device=device)             
            mask = positions.unsqueeze(0).expand(batch_size, max_len-i-1) + i + 1            
            mask = mask < seq_lens.unsqueeze(1) 
            mask = mask.to(torch.float32)
            
            step_loss = -F.log_softmax(torch.cat([positive_pred_i.unsqueeze(-1), neg_pred_i], dim=-1), dim=-1)[:, :, 0]
            step_loss = (step_loss * mask).sum() / mask.sum()
            step_losses.append(step_loss)

        loss = torch.stack(step_losses).mean()
        return loss

    def cpc_accuracy(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)

        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        device = mapped_ctx_embeddings.payload.device

        len_mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
        len_mask = (len_mask < seq_lens.unsqueeze(1).expand(-1, max_seq_len)).float()

        total, accurate = 0, 0
        for i, (positive_pred_i, neg_pred_i) in enumerate(zip(positive_preds, neg_preds)):
            i_mask = len_mask[:, (i + 1):max_seq_len].to(device)
            total += i_mask.sum().item()
            accurate += (((positive_pred_i.unsqueeze(-1).expand(*neg_pred_i.shape) > neg_pred_i) \
                          .sum(dim=-1) == self.n_negatives) * i_mask).sum().item()
        return accurate / total
