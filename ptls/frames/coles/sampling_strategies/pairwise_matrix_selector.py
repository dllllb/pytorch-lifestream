import torch


class PairwiseMatrixSelector:
    """
    Returns matrix with one positive pair on first index and all negative pairs 
    for every possible pair. 
    """
    def get_pair_matrix(self, embeddings, labels):
        n = labels.size(0)

        uniq_num = len(torch.unique(labels))
        split_num = len(labels)//uniq_num

        # construct matrix x, such as x_ij == 1 <==> labels[i] == labels[j]
        x = labels.expand(n, n) - labels.expand(n, n).t()
        x.fill_diagonal_(1)

        indx = torch.where(x == 0)
        positive_pairs = torch.cat((indx[0].reshape(-1, 1), indx[1].reshape(-1, 1)), dim=1)
        positive_pairs = torch.unsqueeze(positive_pairs, 1)

        x.fill_diagonal_(0)
        indx = torch.where(x != 0)

        negative_pairs = torch.cat((indx[0].reshape(-1, 1), indx[1].reshape(-1, 1)), dim=1)
        negative_pairs = negative_pairs.reshape(-1, n - split_num, 2)
        negative_pairs = negative_pairs.repeat(1, split_num - 1, 1)
        negative_pairs = negative_pairs.reshape(-1, n - split_num, 2)

        pairs = torch.cat((positive_pairs, negative_pairs), dim=1)
        return embeddings[pairs]
