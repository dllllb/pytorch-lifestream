import numpy as np

from functools import partial
from multiprocessing import Pool
from sklearn.decomposition import PCA


def quantile_(input_array, q_count):
    return np.quantile(input_array, q=[i / q_count for i in range(1, q_count)], axis=0)


def digitize_(input_array, bins):
    return np.digitize(input_array, bins)


def embeds_quantiles(input_array: np.array, q_count: int = 0, bins: np.array = None, num_workers: int = 0):
    """Quantile-based discretization function.

    Parameters:
    -------
    input_array (np.array): Input array.
    q_count (int): Amount of quantiles. Used only if input parameter `bins` is None.
    bins (np.array):
        If None, then calculate bins as quantiles of input array,
        otherwise only apply bins to input_array. Default: None
    num_workers (int): Amount of workers in Pool in case num_workers>1, without multiprocessing otherwise. Default: 0

    Returns
    -------
    out_array (np.array of ints): discretized input_array
    bins (np.array of floats):
        Returned only if input parameter `bins` is None.
    """

    if bins is None:
        return_bins = True
        reduce_partial = partial(quantile_, q_count=q_count)
        if num_workers > 1:
            with Pool(num_workers) as p:
                bins = p.map(reduce_partial, input_array.T)
            bins = np.concatenate([x.reshape(-1, 1) for x in bins], axis=1)
        else:
            bins = quantile_(input_array, q_count)
    else:
        assert input_array.shape[1] == bins.shape[1]
        return_bins = False
        q_count = bins.shape[0]

    if num_workers > 1:
        with Pool(num_workers) as p:
            out_array = p.starmap(digitize_, zip(input_array.T, bins.T))
    else:
        out_array = map(lambda x, y: np.digitize(x, y), input_array.T, bins.T)
    out_array = np.concatenate([x.reshape(-1, 1) for x in out_array], axis=1)

    if q_count < 128:
        out_array = out_array.astype(np.int8)
    elif q_count < 256:
        out_array -= 128
        out_array = out_array.astype(np.int8)

    if return_bins:
        return out_array, bins
    else:
        return out_array


def reduce_part_(args_list):
    part, n_components, pca = args_list
    if pca is None:
        return_pca = True
        pca = PCA(n_components)
        pca.fit(part)
    else:
        return_pca = False

    transformed = pca.fit_transform(part)
    if return_pca:
        return transformed, pca
    else:
        return transformed


def pca_reduction(input_array, sub_dim: int, n_components: int = 0, pca_list=None, num_workers: int = 0):
    """PCA-based embeddings dimension reduction function.

    Split all array on parts by columns. Reduce each part with PCA. Concatenate results.

    Parameters:
    -------
    input_array (np.array): Input array.
    sub_dim (int): Dimension of each parts.
    n_components (int): Dimension of each parts after reduction. Used only if input parameter `pca_list` is not None.
    pca_list (np.array): list of PCA objects, to be applied to input_array
   num_workers (int): Amount of workers in Pool in case num_workers>1, without multiprocessing otherwise. Default: 0

    Returns
    -------
    out_array (np.array of floats): reduced input_array
    pca_list (list):
        List of PCA objects, fitted on input_array. Returned only if input parameter `pca_list` is None.
    """

    n_parts = (input_array.shape[1] - 1) // sub_dim + 1
    parts = [input_array[:, i * sub_dim: (i + 1) * sub_dim] for i in range(n_parts)]
    if pca_list is None:
        args_list = [(part, n_components, None) for part in parts]
        return_pca = True
    else:
        args_list = [(part, n_components, pca) for part, pca in zip(parts, pca_list)]
        return_pca = False

    if num_workers > 1:
        with Pool(num_workers) as p:
            transformed = p.map(reduce_part_, args_list)
    else:
        transformed = list(map(reduce_part_, args_list))
    if pca_list is None:
        transformed, pca_list = zip(*transformed)
    transformed = np.concatenate(transformed, axis=1)

    if return_pca:
        return transformed, pca_list
    else:
        return transformed
