import torch
from torch import nn


class CLUBLoss(nn.Module):
    def embed_loss_prob(self, pos_probs, neg_probs):
        loss = pos_probs.mean() - neg_probs.mean()
        with torch.no_grad():
            acc = ((pos_probs > 0.5).sum() + (neg_probs < 0.5).sum()) / (pos_probs.shape[0] + neg_probs.shape[0])
            pos_mean = pos_probs.mean()
            pos_std = pos_probs.std()
            pos_delta = pos_probs.max() - pos_probs.min()
            neg_mean = neg_probs.mean()
            neg_std = neg_probs.std()
            neg_delta = neg_probs.max() - neg_probs.min()
        return loss, {"CLUB_embed_loss": loss.item(),
                      "CLUB_accuracy": acc.item(),
                      "CLUB_pos_pred_mean": pos_mean.item(),
                      "CLUB_neg_pred_mean": neg_mean.item(),
                      "CLUB_pos_pred_std": pos_std.item(),
                      "CLUB_neg_pred_std": neg_std.item(),
                      "CLUB_pos_pred_delta": pos_delta.item(),
                      "CLUB_neg_pred_delta": neg_delta.item()}

    def pred_loss_prob(self, pos_probs, neg_probs):
        pos_probs = - pos_probs.mean()
        loss = pos_probs + neg_probs.mean()
        return loss, {"CLUB_prob_loss": pos_probs.item()}
