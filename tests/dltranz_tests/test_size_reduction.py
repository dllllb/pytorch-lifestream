import numpy as np

from ptls.size_reduction import embeds_quantiles, pca_reduction


def test_embeds_quantiles():
    array = np.array([
        [0.1, 0.6, -0.4, 0.9],
        [0.2, 0.7, -0.3, 0.8],
        [0.3, 0.8, -0.2, 0.7],
        [0.4, 0.9, -0.1, 0.6],
        [0.1, 0.6, -0.4, 0.9],
        [0.2, 0.7, -0.3, 0.8],
        [0.3, 0.8, -0.2, 0.7],
        [0.4, 0.9, -0.1, 0.6]
    ])

    out = np.array([
        (0, 0, 0, 3),
        (1, 1, 1, 2),
        (2, 2, 2, 1),
        (3, 3, 3, 0),
        (0, 0, 0, 3),
        (1, 1, 1, 2),
        (2, 2, 2, 1),
        (3, 3, 3, 0),
    ])

    q_embeds, bins = embeds_quantiles(array, q_count=4, num_workers=0)
    assert np.array_equal(q_embeds, out)

    q_embeds = embeds_quantiles(array, bins=bins, num_workers=0)
    assert np.array_equal(q_embeds, out)

    q_embeds, bins = embeds_quantiles(array, q_count=4, num_workers=8)
    assert np.array_equal(q_embeds, out)

    q_embeds = embeds_quantiles(array, bins=bins, num_workers=8)
    assert np.array_equal(q_embeds, out)


def test_pca_reduction():
    array = np.array([
        [0.1,  0.6, -0.4,  0.9],
        [0.2,  0.7, -0.3,  0.8],
        [0.3,  0.8, -0.2,  0.7],
        [0.4,  0.9, -0.1,  0.6],
        [0.1,  0.6, -0.4,  0.9],
        [0.2,  0.7, -0.3,  0.8],
        [0.3,  0.8, -0.2,  0.7],
        [0.4,  0.9, -0.1,  0.6]
    ])

    out = np.array([[-0.3, -0.1,  0.1,  0.3, -0.3, -0.1,  0.1,  0.3]]).T

    reduced_array, pca_list = pca_reduction(array, sub_dim=4, n_components=1, num_workers=0)
    assert np.allclose(reduced_array, out, rtol=1e-07)

    reduced_array = pca_reduction(array, sub_dim=4, pca_list=pca_list, num_workers=0)
    assert np.allclose(reduced_array, out, rtol=1e-07)

    reduced_array, pca_list = pca_reduction(array, sub_dim=4, n_components=1, num_workers=8)
    assert np.allclose(reduced_array, out, rtol=1e-07)

    reduced_array = pca_reduction(array, sub_dim=4, pca_list=pca_list, num_workers=8)
    assert np.allclose(reduced_array, out, rtol=1e-07)
