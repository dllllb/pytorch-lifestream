import torch.nn as nn
import torch
import torch.nn.functional as F


class MSELossBalanced(torch.nn.MSELoss):
    def forward(self, input, target):
        loss = super().forward(input, target)
        loss *= input.size()[1]
        return loss


def outer_pairwise_distance(A, B=None):
    """
        Compute pairwise_distance of Tensors
            A (size(A) = n x d, where n - rows count, d - vector size) and
            B (size(A) = m x d, where m - rows count, d - vector size)
        return matrix C (size n x m), such as C_ij = distance(i-th row matrix A, j-th row matrix B)

        if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if B is None: B = A

    max_size = 2 ** 26
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)

    if n * m * d <= max_size or m == 1:

        return torch.pairwise_distance(
            A[:, None].expand(n, m, d).reshape((-1, d)),
            B.expand(n, m, d).reshape((-1, d))
        ).reshape((n, m))

    else:

        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_pairwise_distance(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, margin, pair_selector):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)

        pos_dist = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]])
        neg_dist = F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])

        positive_loss = pos_dist.pow(2)
        negative_loss = F.relu(self.margin - neg_dist).pow(2)

        return {
            'loss_pos': positive_loss.sum(),
            'loss_neg': negative_loss.sum(),
            'dist_pos_mean': pos_dist.mean(),
            'dist_neg_mean': neg_dist.mean(),
            'dist_pos_max': pos_dist.max(),
            'dist_neg_min': neg_dist.min(),
        }


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class HardNegativePairSelector(PairSelector):
    """
    Generates all possible possitive pairs given labels and
         neg_count hardest negative example for each example
    """

    def __init__(self, neg_count=1):
        super(HardNegativePairSelector, self).__init__()
        self.neg_count = neg_count

    def get_pairs(self, embeddings, labels):
        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        n = labels.size(0)
        x = labels.expand(n, n) - labels.expand(n, n).t()

        # positive pairs
        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero()

        # hard negative minning
        mat_distances = outer_pairwise_distance(embeddings.detach())  # pairwise_distance

        upper_bound = int((2 * n) ** 0.5) + 1
        mat_distances = ((upper_bound - mat_distances) * (x != 0).type(
            mat_distances.dtype))  # filter: get only negative pairs

        values, indices = mat_distances.topk(k=self.neg_count, dim=0, largest=True)
        negative_pairs = torch.stack([
            torch.arange(0, n, dtype=indices.dtype, device=indices.device).repeat(self.neg_count),
            torch.cat(indices.unbind(dim=0))
        ]).t()

        return positive_pairs, negative_pairs


class Accuracy(torch.nn.Module):
    def forward(self, output, target):
        t = (output > 0.5).to(dtype=target.dtype)
        t = (target == t).to(dtype=target.dtype)
        return t.mean()
