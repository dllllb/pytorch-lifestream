import torch

from ptls.metric_learn.sampling_strategies import AllPositivePairSelector, HardNegativePairSelector, \
    DistanceWeightedPairSelector


def get_data():
    b, c, h = 3, 2, 2  # Batch, num Classes, Hidden size
    x = torch.tensor([[0., 1.],
                      [1., 0.],
                      [1., 1.],
                      [0.1, 0.9],
                      [0.8, 0.2],
                      [1.0, 0.9]])

    y = torch.arange(c).view(-1, 1).expand(c, b).reshape(-1)
    return x, y


def check_positive_pairs(positive_pairs, y):
    # positives from same classes
    assert torch.equal(y[positive_pairs[:, 0]], y[positive_pairs[:, 1]])


def check_negative_pairs(negative_pairs, y):
    # positives from different classes
    assert torch.all(~torch.eq(y[negative_pairs[:, 0]], y[negative_pairs[:, 1]]))


def test_all_pair_selector():
    x, y = get_data()
    sampling_strategy = AllPositivePairSelector(balance=False)

    positive_pairs, negative_pairs = sampling_strategy.get_pairs(x, y)
    check_positive_pairs(positive_pairs, y)
    check_negative_pairs(negative_pairs, y)

    true_positive_pairs = torch.LongTensor([[0, 1],
                                            [0, 2],
                                            [1, 2],
                                            [3, 4],
                                            [3, 5],
                                            [4, 5]])

    true_negative_pairs = torch.LongTensor([[0, 3],
                                            [0, 4],
                                            [0, 5],
                                            [1, 3],
                                            [1, 4],
                                            [1, 5],
                                            [2, 3],
                                            [2, 4],
                                            [2, 5]])

    assert torch.equal(positive_pairs, true_positive_pairs)
    assert torch.equal(negative_pairs, true_negative_pairs)


def test_hard_pair_selector():
    x, y = get_data()
    sampling_strategy = HardNegativePairSelector()

    positive_pairs, negative_pairs = sampling_strategy.get_pairs(x, y)
    check_positive_pairs(positive_pairs, y)
    check_negative_pairs(negative_pairs, y)

    true_positive_pairs = torch.LongTensor([[0, 1],
                                            [0, 2],
                                            [1, 2],
                                            [3, 4],
                                            [3, 5],
                                            [4, 5]])

    true_negative_pairs = torch.LongTensor([[0, 3],
                                            [1, 4],
                                            [2, 5],
                                            [3, 0],
                                            [4, 1],
                                            [5, 2]])

    assert torch.equal(positive_pairs, true_positive_pairs)
    assert torch.equal(negative_pairs, true_negative_pairs)


def test_distance_weighted_pair_selector():
    x, y = get_data()
    sampling_strategy = DistanceWeightedPairSelector(batch_k=3)
    positive_pairs, negative_pairs = sampling_strategy.get_pairs(x, y)
    check_positive_pairs(positive_pairs, y)
    check_negative_pairs(negative_pairs, y)
