import torch

from ptls.frames.coles.sampling_strategies import MatrixMasker, PairwiseMatrixSelector


def get_data():
    b, c, h = 3, 2, 1  # Batch, num Classes, Hidden size
    x = torch.tensor([[0.0],
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0]])

    y = torch.arange(c).view(-1, 1).expand(c, b).reshape(-1)
    return x, y


def test_matrix_masker():
    b, c, h = 3, 3, 2  # Batch, num Classes, Hidden size
    x = torch.ones((9, 9))
    y = torch.arange(c).view(-1, 1).expand(c, b).reshape(-1)

    true_matrix = torch.tensor(
        [[0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0]]
    )

    masker = MatrixMasker()
    masked_matrix = masker.get_masked_matrix(x, y)

    assert torch.equal(masked_matrix, true_matrix)
    assert type(masked_matrix) is torch.Tensor


def test_pairwise_matrix_selector():
    x, y = get_data()

    true_pair_matrix = torch.tensor([
        [[[0.],[1.]], [[0.], [3.]], [[0.],[4.]], [[0.],[5.]]],
        [[[0.],[2.]], [[0.], [3.]], [[0.],[4.]], [[0.],[5.]]],
        [[[1.],[0.]], [[1.], [3.]], [[1.], [4.]], [[1.], [5.]]],
        [[[1.],[2.]], [[1.], [3.]], [[1.],[4.]], [[1.], [5.]]],
        [[[2.], [0.]], [[2.], [3.]], [[2.], [4.]], [[2.], [5.]]],
        [[[2.], [1.]], [[2.], [3.]], [[2.], [4.]], [[2.], [5.]]],
        [[[3.], [4.]], [[3.], [0.]], [[3.], [1.]], [[3.], [2.]]],
        [[[3.], [5.]], [[3.], [0.]], [[3.], [1.]], [[3.], [2.]]],
        [[[4.], [3.]], [[4.], [0.]], [[4.], [1.]], [[4.], [2.]]],
        [[[4.], [5.]], [[4.], [0.]], [[4.], [1.]], [[4.], [2.]]],
        [[[5.], [3.]], [[5.], [0.]], [[5.], [1.]], [[5.], [2.]]],
        [[[5.], [4.]], [[5.], [0.]], [[5.], [1.]], [[5.], [2.]]]])

    pair_selector = PairwiseMatrixSelector()
    pair_matrix = pair_selector.get_pair_matrix(x, y)

    assert torch.equal(pair_matrix, true_pair_matrix)
    assert type(pair_matrix) is torch.Tensor
