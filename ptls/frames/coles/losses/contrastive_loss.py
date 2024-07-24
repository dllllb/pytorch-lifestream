import torch
from torch import nn as nn
from torch.nn import functional as F
from ptls.nn.normalization import L2NormEncoder
from itertools import chain
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, margin, sampling_strategy):
        super().__init__()
        self.margin = margin
        self.pair_selector = sampling_strategy
        self.loss_name = "coles"

    def forward(self, embeddings, target, *argv):
        if type(target) is dict:
            target = target["coles_target"]

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)
        #positive_loss = - (embeddings[positive_pairs[:, 0]] * embeddings[positive_pairs[:, 1]]).sum(axis=-1).mean()

        negative_loss = F.relu(
            self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2)
        #negative_loss = F.relu(
        #    1 + (embeddings[negative_pairs[:, 0]] * embeddings[negative_pairs[:, 1]]).sum(axis=-1) - self.margin
        #).mean()
        loss = torch.cat([positive_loss, negative_loss], dim=0).mean()
        #loss = positive_loss + negative_loss

        with torch.no_grad():
            rand_inds = torch.randperm(embeddings.shape[0])
            rand_embs = embeddings[rand_inds]
            rand_cos = (embeddings * rand_embs).sum(dim=-1).mean()
            p_l = positive_loss.mean()
            n_l = negative_loss.mean()

        return loss, {"COLES_loss_0": loss.item(),
                      "COLES_pos_loss_0": p_l.item(),
                      "COLES_neg_loss_0": n_l.item(),
                      "COLES_cos_b2neg_b": rand_cos.item()}


class MultiContrastiveLoss(nn.Module):
    def __init__(self, margin, sampling_strategy):
        super().__init__()
        self.margin = margin
        self.pair_selector = sampling_strategy
        self.loss_name = "coles"

    def forward(self, multi_embeddings, target, *argv):
        if type(target) is dict:
            target = target["coles_target"]

        loss = 0
        info = dict()
        for i, embeddings in enumerate(multi_embeddings):
            positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
            positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]],
                                                embeddings[positive_pairs[:, 1]]).pow(2)
            #positive_loss = - (embeddings[positive_pairs[:, 0]] * embeddings[positive_pairs[:, 1]]).sum(axis=-1)

            negative_loss = F.relu(self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]],
                                                                     embeddings[negative_pairs[:, 1]])).pow(2)
            #negative_loss = F.relu(
            #    1 + (embeddings[negative_pairs[:, 0]] * embeddings[negative_pairs[:, 1]]).sum(axis=-1) - self.margin
            #)
            loss_i = torch.cat([positive_loss, negative_loss], dim=0).mean()
            loss += loss_i
            with torch.no_grad():
                p_l = positive_loss.mean()
                n_l = negative_loss.mean()

            info["COLES_loss_" + str(i)] = loss_i.item()
            info["COLES_pos_loss_" + str(i)] = p_l.item()
            info["COLES_neg_loss_" + str(i)] = n_l.item()
        loss = loss / len(multi_embeddings)
        info["COLES_mean_loss"] = loss.item()
        return loss, info


class CLUBLoss(nn.Module):
    def __init__(self, log_var=-3, emb_coef=1., prob_coef=1.):
        super().__init__()
        self.emb_coef = emb_coef
        self.prob_coef = prob_coef
        self.register_buffer('log_var', torch.tensor(log_var))

    def get_mu_log_var(self, inp):
        mu = self.model_mu(inp)
        log_var = self.model_log_var(inp) * 2
        return mu, log_var

    def embed_loss(self, vec_true, vec_pred):
        random_inds = torch.randperm(vec_true.shape[0])
        neg_vec_true = vec_true[random_inds]
        pos_probs = (-(vec_true - vec_pred) ** 2 / self.log_var.exp() - self.log_var).sum(dim=-1)
        neg_probs = (-(neg_vec_true - vec_pred) ** 2 / self.log_var.exp() - self.log_var).sum(dim=-1)
        loss = (pos_probs - neg_probs).mean()
        with torch.no_grad():
            cos_random_dist = (vec_pred * neg_vec_true).sum(dim=-1).mean()
            cos_b2neg_b = (vec_true * neg_vec_true).sum(dim=-1).mean()
            cos_mu2neg_mu = (vec_pred * vec_pred[random_inds]).sum(dim=-1).mean()
            cos_dist = (vec_pred * vec_true).sum(dim=-1).mean()
        return loss, {"CLUB_embed_loss": loss.item(),
                      "CLUB_cos_random_dist": cos_random_dist.item(),
                      "CLUB_cos_b2neg_b": cos_b2neg_b.item(),
                      "CLUB_cos_mu2neg_mu": cos_mu2neg_mu.item(),
                      "CLUB_cos_dist": cos_dist.item()}

    def pred_loss(self, vec_true, vec_pred):
        loss = ((vec_true - vec_pred) ** 2 / self.log_var.exp() + self.log_var).sum(dim=-1).mean()
        return loss, {"CLUB_prob_loss": loss.item()}

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
        pl = - pos_probs.mean()
        loss = pl + neg_probs.mean()
        return loss, {"CLUB_prob_loss": pl.item()}

    def forward(self, *argv):
        return
