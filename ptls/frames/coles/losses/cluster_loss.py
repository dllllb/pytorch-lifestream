import torch
from torch import nn


class ClusterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.loss_name = "cluster"

    def forward(self, embs, idx, clusters):
        labels = list()
        logits = list()
        num_clusters = list()

        if type(idx) is dict:
            idx = idx["cluster_target"]

        for n, (idx2cluster, prototypes, density) in enumerate(
                zip(clusters['idx2cluster'], clusters['centroids'], clusters['density'])):
            clus_logits = torch.mm(embs, prototypes.t()) / density
            clus_labels = idx2cluster[idx]

            labels.append(clus_labels)
            logits.append(clus_logits)
            num_clusters.append(len(prototypes))

        losses = list()
        loss = 0
        for log, lab in zip(logits, labels):
            temp_loss = self.criterion(log, lab)
            loss += temp_loss
            losses.append(temp_loss.detach().cpu().item())
        loss = loss / len(labels)
        return loss, {'name': self.loss_name,
                      'num_cluster': num_clusters,
                      'logits': logits,
                      'labels': labels,
                      'losses': losses}
