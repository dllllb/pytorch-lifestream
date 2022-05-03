import torch

from ptls.metric_learn.sampling_strategies import AllTripletSelector, RandomNegativeTripletSelector, \
    HardTripletSelector, SemiHardTripletSelector


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


def check_triplets(triplets, y):
    # positives from same classes as anchor elements
    assert torch.equal(y[triplets[:, 0]], y[triplets[:, 1]])

    # positives from another classes than anchor elements
    assert torch.all(~torch.eq(y[triplets[:, 0]], y[triplets[:, 2]]))


def test_random_triplet_selector():
    x, y = get_data()
    sampling_strategy = RandomNegativeTripletSelector()

    triplets = sampling_strategy.get_triplets(x, y)
    check_triplets(triplets, y)


def test_all_triplet_selector():
    x, y = get_data()
    sampling_strategy = AllTripletSelector()

    triplets = sampling_strategy.get_triplets(x, y)
    check_triplets(triplets, y)

    true_triplets = torch.LongTensor([[0, 1, 3],
                                      [0, 1, 4],
                                      [0, 1, 5],
                                      [0, 2, 3],
                                      [0, 2, 4],
                                      [0, 2, 5],
                                      [1, 2, 3],
                                      [1, 2, 4],
                                      [1, 2, 5],
                                      [3, 4, 0],
                                      [3, 4, 1],
                                      [3, 4, 2],
                                      [3, 5, 0],
                                      [3, 5, 1],
                                      [3, 5, 2],
                                      [4, 5, 0],
                                      [4, 5, 1],
                                      [4, 5, 2]])

    assert torch.equal(triplets, true_triplets)


def test_hard_triplet_selector():
    x, y = get_data()
    sampling_strategy = HardTripletSelector()

    triplets = sampling_strategy.get_triplets(x, y)
    check_triplets(triplets, y)

    true_triplets = torch.LongTensor([[0, 1, 3],
                                      [0, 2, 3],
                                      [1, 2, 4],
                                      [3, 4, 0],
                                      [3, 5, 0],
                                      [4, 5, 1]])

    assert torch.equal(triplets, true_triplets)


def test_semihard_triplet_selector():
    x, y = get_data()
    sampling_strategy = SemiHardTripletSelector()

    triplets = sampling_strategy.get_triplets(x, y)
    check_triplets(triplets, y)

    true_triplets = torch.LongTensor([[0, 1, 4],
                                      [0, 2, 5],
                                      [1, 2, 3],
                                      [3, 4, 1],
                                      [3, 5, 2],
                                      [4, 5, 2]])

    assert torch.equal(triplets, true_triplets)
