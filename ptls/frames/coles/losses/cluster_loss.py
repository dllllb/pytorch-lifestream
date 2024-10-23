import torch
from torch import nn
from itertools import chain
from collections import defaultdict


class ClusterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embs, idx, clusters):
        labels = list()
        logits = list()
        num_clusters = list()

        if type(idx) is dict:
            idx = idx["cluster_target"]

        for idx2cluster, prototypes, density in zip(clusters['idx2cluster'],
                                                    clusters['centroids'],
                                                    clusters['density']):
            clus_logits = torch.mm(embs, prototypes.t()) / density
            clus_labels = idx2cluster[idx]

            labels.append(clus_labels)
            logits.append(clus_logits)
            num_clusters.append(len(prototypes))

        logits, libels = list(), list()
        info = dict()
        loss = 0
        for n, log, lab in zip(num_clusters, logits, labels):
            loss_ = self.criterion(log, lab)
            loss += loss_
            info[f'CLUSTER_{n}_loss'] = loss_.item()
            info[f'CLUSTER_{n}_logits_labels'] = (log.detach().cpu().numpy(), lab.detach().cpu().numpy())
        loss = loss / len(labels)
        return loss, info


class ClusterAndContrastive(nn.Module):
    def __init__(self, cluster_loss, contrastive_loss, cluster_weight=1., contrastive_weight=1.):
        super().__init__()
        self.cluster_loss = cluster_loss
        self.contrastive_loss = contrastive_loss
        self.cluster_weight = cluster_weight
        self.contrastive_weight = contrastive_weight

    def forward(self, embs, idx, clusters):
        cluster_target, contrastive_target = idx["cluster_target"], idx["coles_target"]
        cluster_loss, cluster_info = self.cluster_loss(embs, cluster_target, clusters)
        contrastive_loss, contrastive_info = self.contrastive_loss(embs, contrastive_target)

        loss = self.cluster_weight * cluster_loss + self.contrastive_weight * contrastive_loss
        info = {k: v for k, v in chain(cluster_info.items(), contrastive_info.items())}
        return loss, info
