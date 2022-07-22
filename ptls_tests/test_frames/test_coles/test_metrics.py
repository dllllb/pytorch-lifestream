import math
import torch

from ptls.frames.coles.metric import outer_cosine_similarity, outer_pairwise_distance, metric_recall_top_K, \
    BatchRecallTopK


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
    x = torch.tensor([[0., 1.],       # 0
                      [0., 0.9],      # 0
                      [0.95, 0.],     # 0
                      [1., 0.],       # 1
                      [0.9, 0.],      # 1
                      [0.0, 0.95]])   # 1

    y = torch.arange(c).view(-1, 1).expand(c, b).reshape(-1)
    return x, y


def test_metric_recall_top_k():
    x, y = get_ml_data()
    metric = metric_recall_top_K(x, y, K=3 - 1, metric='euclidean')
    true_value = 1/3
    assert abs(metric - true_value) < 1e-6


def test_metric_recall_top_k_small_batch():
    x, y = get_ml_data()
    metric = metric_recall_top_K(x[1:5], y[1:5], K=3 - 1, metric='euclidean')
    true_value = 3 / 8
    assert abs(metric - true_value) < 1e-6


def test_batch_recall_top():
    x, y = get_ml_data()
    metric = BatchRecallTopK(K=3 - 1, metric='euclidean')
    metric(x, y)
    res = metric.compute()
    true_value = 1 / 3
    assert abs(res - true_value) < 1e-6


def test_batch_recall_top2_steps():
    x, y = get_ml_data()
    metric = BatchRecallTopK(K=3 - 1, metric='euclidean')
    metric(x, y)
    metric(x[1:5], y[1:5])
    res = metric.compute()
    true_value = 17 / 48
    assert abs(res - true_value) < 1e-6
