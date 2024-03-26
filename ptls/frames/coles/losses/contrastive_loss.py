import torch
from torch import nn as nn
from torch.nn import functional as F


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
    def __init__(self, input_dim, h_sizes, emb_coef=1., prob_coef=1.):
        super().__init__()
        self.input_dim = input_dim
        h_sizes = [input_dim * 2] + h_sizes + [1]
        layers = list()
        for i in range(1, len(h_sizes)):
            layers.append(nn.Linear(h_sizes[i-1], h_sizes[i]))
            layers.append(nn.BatchNorm1d(h_sizes[i]))
            layers.append(nn.ReLU())
        layers.pop(-1)
        layers.append(nn.Sigmoid())
        self.prob_model = nn.Sequential(*layers)
        self.emb_coef = emb_coef
        self.prob_coef = prob_coef

    def switch_prob_model_rg(self, true_false):
        for param in self.prob_model.parameters():
            param.requires_grad = true_false

    def forward(self, multi_embeddings, *argv):
        assert len(multi_embeddings) == 2
        domain_a, domain_b = multi_embeddings
        random_inds = torch.randperm(domain_a.shape[0])
        neg_inp = torch.concatenate([domain_a[random_inds], domain_b], dim=-1)
        pos_inp = torch.concatenate([domain_a, domain_b], dim=-1)

        self.switch_prob_model_rg(False)
        embed_model_pos = self.prob_model(pos_inp)
        embed_model_neg = self.prob_model(neg_inp)
        embed_model_loss = (torch.log(embed_model_pos) - torch.log(embed_model_neg)).mean()
        with torch.no_grad():
            accuracy = ((embed_model_pos > 0.5).float().mean() + (embed_model_neg < 0.5).float().mean()) / 2

        self.switch_prob_model_rg(True)
        prob_model_loss = self.prob_model(pos_inp.detach())
        prob_model_loss = -torch.log(prob_model_loss).mean()

        loss = self.emb_coef * embed_model_loss + self.prob_coef * prob_model_loss
        info = {"CLUB_embed_loss": embed_model_loss.item(),
                "CLUB_prob_loss": prob_model_loss.item(),
                "CLUB_total_loss": loss.item(),
                "CLUB_accuracy": accuracy.item()}
        return loss, info
