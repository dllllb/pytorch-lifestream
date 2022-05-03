from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

from .metric import outer_pairwise_distance


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


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """

    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        n = labels.size(0)
        x = labels.expand(n, n) - labels.expand(n, n).t()

        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)
        negative_pairs = torch.triu((x != 0).int(), diagonal=1).nonzero(as_tuple=False)

        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


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
        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)
        
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


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        np_labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(np_labels):
            label_mask = (np_labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets)).to(labels.device)


class RandomNegativeTripletSelector(TripletSelector):
    """
        Generate triplets with all positive pairs and random negative example for each anchor
    """

    def __init__(self, neg_count=1):
        super(RandomNegativeTripletSelector, self).__init__()
        self.neg_count = neg_count

    def get_triplets(self, embeddings, labels):
        n = labels.size(0)

        # construct matrix x, such as x_ij == 1 <==> labels[i] == labels[j]
        x = labels.expand(n, n) - labels.expand(n, n).t()

        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        m = positive_pairs.size(0)
        anchor_labels = labels[positive_pairs[:, 0]]

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = (labels.expand(m, n) == anchor_labels.expand(n, m).t())

        negative_pairs = (x == 0).type(embeddings.dtype)
        negative_pairs_prob = (negative_pairs.t() / negative_pairs.sum(dim=1)).t()
        negative_pairs_rand = torch.multinomial(negative_pairs_prob, 1)

        triplets = torch.cat([positive_pairs, negative_pairs_rand], dim=1)

        return triplets


class HardTripletSelector(TripletSelector):
    """
        Generate triplets with all positive pairs and the neg_count hardest negative example for each anchor
    """

    def __init__(self, neg_count=1):
        super(HardTripletSelector, self).__init__()
        self.neg_count = neg_count

    def get_triplets(self, embeddings, labels):
        n = labels.size(0)

        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        x = labels.expand(n, n) - labels.expand(n, n).t()

        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        m = positive_pairs.size(0)

        anchor_embed = embeddings[positive_pairs[:, 0]].detach()
        anchor_labels = labels[positive_pairs[:, 0]]

        # pos_embed = embeddings[positive_pairs[:,0]].detach()

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = (labels.expand(m, n) == anchor_labels.expand(n, m).t())

        mat_distances = outer_pairwise_distance(anchor_embed, embeddings.detach())  # pairwise_distance anchors x all

        upper_bound = int((2 * n) ** 0.5) + 1
        mat_distances = ((upper_bound - mat_distances) * (x == 0).type(
            mat_distances.dtype))  # filter: get only negative pairs

        values, indices = mat_distances.topk(k=self.neg_count, dim=1, largest=True)

        triplets = torch.cat([
            positive_pairs.repeat(self.neg_count, 1),
            torch.cat(indices.unbind(dim=0)).view(-1, 1)
        ], dim=1)

        return triplets


class SemiHardTripletSelector(TripletSelector):
    """
        Generate triplets with semihard sampling strategy

        "FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR 2015
        https://arxiv.org/abs/1503.03832
    """

    def __init__(self, neg_count=1):
        super(SemiHardTripletSelector, self).__init__()
        self.neg_count = neg_count

    def get_triplets(self, embeddings, labels):
        n = labels.size(0)

        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        x = labels.expand(n, n) - labels.expand(n, n).t()

        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        m = positive_pairs.size(0)

        anchor_embed = embeddings[positive_pairs[:, 0]].detach()
        anchor_labels = labels[positive_pairs[:, 0]]

        pos_embed = embeddings[positive_pairs[:, 1]].detach()

        D_ap = F.pairwise_distance(anchor_embed, pos_embed)

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = (labels.expand(m, n) == anchor_labels.expand(n, m).t())

        mat_distances = outer_pairwise_distance(anchor_embed, embeddings.detach())  # pairwise_distance anchors x all

        neg_mat_distances = mat_distances * (x == 0).type(mat_distances.dtype)  # filter: get only negative pairs

        # negatives_outside: smallest D_an where D_an > D_ap.
        upper_bound = int((2 * n) ** 0.5) + 1
        negatives_outside = (upper_bound - neg_mat_distances) * \
                            (neg_mat_distances > D_ap.expand(n, m).t()).type(neg_mat_distances.dtype)
        values, negatives_outside = negatives_outside.topk(k=1, dim=1, largest=True)

        # negatives_inside: largest D_an
        values, negatives_inside = neg_mat_distances.topk(k=1, dim=1, largest=True)

        # whether exist negative n, such that D_an > D_ap.
        semihard_exist = ((neg_mat_distances > D_ap.expand(n, m).t()).sum(dim=1) > 0).view(-1, 1)

        negatives_indeces = torch.where(semihard_exist, negatives_outside, negatives_inside)

        triplets = torch.cat([positive_pairs, negatives_indeces], dim=1)

        return triplets


def get_sampling_strategy(params):
    if params['train.sampling_strategy'] is None:
        sampling_strategy = None

    elif params['train.sampling_strategy'] == 'AllPositivePair':
        sampling_strategy = AllPositivePairSelector()

    elif params['train.sampling_strategy'] == 'HardNegativePair':
        kwargs = {
            'neg_count': params.get('train.neg_count', None),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        sampling_strategy = HardNegativePairSelector(**kwargs)

    elif params['train.sampling_strategy'] == 'AllTriplets':
        sampling_strategy = AllTripletSelector()

    elif params['train.sampling_strategy'] == 'RandomNegativeTriplets':
        sampling_strategy = RandomNegativeTripletSelector()

    elif params['train.sampling_strategy'] == 'HardTriplets':
        kwargs = {
            'neg_count': params.get('train.neg_count', None),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        sampling_strategy = HardTripletSelector(**kwargs)

    elif params['train.sampling_strategy'] == 'SemiHardTriplets':
        sampling_strategy = SemiHardTripletSelector()

    elif params['train.sampling_strategy'] == 'DistanceWeightedPair':
        kwargs = {
            'batch_k': params['train.n_samples_from_class'],
            'cutoff': params.get('train.cutoff', None),
            'nonzero_loss_cutoff': params.get('train.nonzero_loss_cutoff', None),
            'normalize': params.get('train.normalize', None),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        sampling_strategy = DistanceWeightedPairSelector(**kwargs)

    else:
        raise AttributeError(f'wrong sampling_strategy "{params["train.sampling_strategy"]}"')

    return sampling_strategy
