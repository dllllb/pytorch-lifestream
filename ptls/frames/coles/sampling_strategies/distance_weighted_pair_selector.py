import numpy as np
import torch

from ptls.frames.coles.sampling_strategies.pair_selector import PairSelector
from ptls.frames.coles.metric import outer_pairwise_distance


class DistanceWeightedPairSelector(PairSelector):
    """
    Distance Weighted Sampling

    "Sampling Matters in Deep Embedding Learning", ICCV 2017
    https://arxiv.org/abs/1706.07567
    code based on https://github.com/suruoxi/DistanceWeightedSampling

    Generates pairs correspond to distances

    parameters
    ----------
    batch_k: int
        number of images per class

    Inputs:
        data: input tensor with shape (batch_size, embed_dim)
            Here we assume the consecutive batch_k examples are of the same class.
            For example, if batch_k = 5, the first 5 examples belong to the same class,
            6th-10th examples belong to another class, etc.
    Outputs:
        a_indices: indicess of anchors
        x[a_indices]
        x[p_indices]
        x[n_indices]
        xxx

    """

    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize=False):
        super(DistanceWeightedPairSelector, self).__init__()
        self.batch_k = batch_k
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.normalize = normalize

    def get_pairs(self, x, labels):
        k = self.batch_k
        n, d = x.shape
        distance = outer_pairwise_distance(x.detach())
        distance = distance.clamp(min=self.cutoff)
        log_weights = ((2.0 - float(d)) * distance.log() - (float(d - 3) / 2) * torch.log(
            torch.clamp(1.0 - 0.25 * (distance * distance), min=1e-8)))

        if self.normalize:
            log_weights = (log_weights - log_weights.min()) / (log_weights.max() - log_weights.min() + 1e-8)

        weights = torch.exp(log_weights - torch.max(log_weights))

        device = x.device
        weights = weights.to(device)

        mask = torch.ones_like(weights)
        for i in range(0, n, k):
            mask[i:i + k, i:i + k] = 0

        mask_uniform_probs = mask.double() * (1.0 / (n - k))

        weights = weights * mask * ((distance < self.nonzero_loss_cutoff).float()) + 1e-8
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        weights = weights / weights_sum

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.cpu().numpy()
        for i in range(n):
            block_idx = i // k

            if weights_sum[i] != 0:
                n_indices += np.random.choice(n, k - 1, p=np_weights[i]).tolist()
            else:
                n_indices += np.random.choice(n, k - 1, p=mask_uniform_probs[i]).tolist()
            for j in range(block_idx * k, (block_idx + 1) * k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        positive_pairs = [[a, p] for a, p in zip(a_indices, p_indices)]
        negative_pairs = [[a, n] for a, n in zip(a_indices, n_indices)]

        return torch.LongTensor(positive_pairs).to(device), torch.LongTensor(negative_pairs).to(device)