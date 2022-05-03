import math
import torch

from ptls.metric_learn.metric import outer_cosine_similarity, outer_pairwise_distance, metric_Recall_top_K, \
    BatchRecallTopPL


def test_outer_cosine_similarity1():
    x = torch.eye(10, dtype=torch.float32)

    dists = outer_cosine_similarity(x, x)
    true_dists = torch.eye(10, dtype=torch.float32)

    assert torch.allclose(dists, true_dists)


def test_outer_cosine_similarity2():
    fi = 1
    x = torch.tensor([
        [math.cos(fi), -math.sin(fi)],
        [math.sin(fi), math.cos(fi)]
    ])

    dists = outer_cosine_similarity(x, x)
    true_dists = torch.eye(2, dtype=torch.float32)

    assert torch.allclose(dists, true_dists)


def test_outer_pairwise_distance1():
    x = torch.eye(10, dtype=torch.float32)

    dists = outer_pairwise_distance(x, x)
    true_dists = (1 - torch.eye(10, dtype=torch.float32)) * math.sqrt(2)

    assert torch.allclose(dists, true_dists, atol=1e-5)


def test_outer_pairwise_distance2():
    fi = 1
    x = torch.tensor([
        [math.cos(fi), -math.sin(fi)],
        [math.sin(fi), math.cos(fi)]
    ])

    dists = outer_pairwise_distance(x, x)
    true_dists = (1 - torch.eye(2, dtype=torch.float32)) * math.sqrt(2)

    assert torch.allclose(dists, true_dists, atol=1e-5)


def get_ml_data():
    b, c, h = 3, 2, 2  # Batch, num Classes, Hidden size
    x = torch.tensor([[0., 1.],
                      [0., 0.9],
                      [0.95, 0.],
                      [1., 0.],
                      [0.9, 0.],
                      [0.0, 0.95]])

    y = torch.arange(c).view(-1, 1).expand(c, b).reshape(-1)
    return x, y


def test_metric_recall_top_k():
    x, y = get_ml_data()
    metric = metric_Recall_top_K(x, y, K=2, metric='euclidean')
    true_value = 1/3
    assert abs(metric - true_value) < 1e-6


def test_batch_recall_top():
    x, y = get_ml_data()
    metric = BatchRecallTopPL(K=2, metric='euclidean')
    metric(x, y)
    res = metric.compute()
    true_value = 1 / 3
    assert abs(res - true_value) < 1e-6
