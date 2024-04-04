import torch
from torch import nn as nn
from torch.nn import functional as F
from ptls.nn.normalization import L2NormEncoder
from itertools import chain


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

        negative_loss = F.relu(
            self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0).mean()

        return loss, {"loss_0": loss.cpu().item()}


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

            negative_loss = F.relu(self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]],
                                                                     embeddings[negative_pairs[:, 1]])).pow(2)
            loss_i = torch.cat([positive_loss, negative_loss], dim=0).mean()
            loss += loss_i
            info["COLES_loss_" + str(i)] = loss_i.item()
        loss = loss / len(multi_embeddings)
        info["COLES_mean_loss"] = loss.item()
        return loss, info


class CLUBLoss(nn.Module):
    def __init__(self, input_dim, h_sizes, emb_coef=1., prob_coef=1., use_batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        h_sizes = [input_dim] + h_sizes + [input_dim]
        layers_mu, layers_log_var = list(), list()

        for i in range(1, len(h_sizes)):

            layers_mu.append(nn.Linear(h_sizes[i-1], h_sizes[i]))
            if use_batch_norm:
                layers_mu.append(nn.BatchNorm1d(h_sizes[i]))
            layers_mu.append(nn.ReLU())

            layers_log_var.append(nn.Linear(h_sizes[i - 1], h_sizes[i]))
            if use_batch_norm:
                layers_log_var.append(nn.BatchNorm1d(h_sizes[i]))
            layers_log_var.append(nn.ReLU())

        layers_mu.pop(-1)
        if use_batch_norm:
            layers_mu.pop(-1)
        layers_mu.append(L2NormEncoder())

        layers_log_var.pop(-1)
        if use_batch_norm:
            layers_log_var.pop(-1)
        layers_log_var.append(nn.Tanh())

        self.model_mu = nn.Sequential(*layers_mu)
        self.model_log_var = nn.Sequential(*layers_log_var)
        self.emb_coef = emb_coef
        self.prob_coef = prob_coef

    def switch_prob_model_rg(self, true_false):
        for param in chain(self.model_mu.parameters(), self.model_log_var.parameters()):
            param.requires_grad = true_false

    def get_mu_log_var(self, inp):
        mu = self.model_mu(inp)
        log_var = self.model_log_var(inp)
        return mu, log_var

    def forward(self, multi_embeddings, *argv):
        assert len(multi_embeddings) == 2
        domain_a, domain_b = multi_embeddings

        self.switch_prob_model_rg(False)
        mu, log_var = self.get_mu_log_var(domain_a)
        pos_probs = (-(domain_b - mu) ** 2 / log_var.exp() - log_var).sum(dim=-1)
        neg_probs = ((-(domain_b.unsqueeze(0) - mu.unsqueeze(1)) ** 2).mean(dim=1)
                     / log_var.exp() - log_var).sum(dim=-1)
        embed_model_loss = (pos_probs - neg_probs).mean()

        self.switch_prob_model_rg(True)
        mu, log_var = self.get_mu_log_var(domain_a.detach())
        prob_model_loss = ((domain_b.detach() - mu) ** 2 / log_var.exp() + log_var).sum(dim=-1).mean()

        with torch.no_grad():
            cos_dist = (mu * domain_b).sum(dim=-1).mean()

        loss = self.emb_coef * embed_model_loss + self.prob_coef * prob_model_loss
        info = {"CLUB_embed_loss": embed_model_loss.item(),
                "CLUB_prob_loss": prob_model_loss.item(),
                "CLUB_total_loss": loss.item(),
                "CLUB_cos_dist": cos_dist.item()}
        return loss, info
