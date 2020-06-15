import numpy as np

from sklearn.decomposition import PCA
from multiprocessing import Pool


def embeds_quantiles(input_array: np.array, q_count: int = 0, bins: np.array = None):
    """Quantile-based discretization function.  

    Parameters:
    -------
    input_array (np.array): Input array.
    q_count (int): Amount of quantiles. Used only if input parameter `bins` is None.
    quantiles (np.array): 
        If None, then calculate bins as quantiles of input array, 
        otherwise only apply bins to input_array. Default: None

    Returns
    -------
    out_array (np.array of ints): discretized input_array
    bins (np.array of floats):
        Returned only if input parameter `bins` is None.
    """

    if bins is None:
        return_bins = True
        bins = np.quantile(input_array, q=[i / q_count for i in range(1, q_count)], axis=0)
    else:
        return_bins = False
    result = []
    for i in range(input_array.shape[1]):
        result.append(np.digitize(input_array[:, i], bins[:, i]).reshape(-1, 1))
    out_array = np.concatenate(result, axis=1)

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


def pca_reduction(input_array, sub_dim: int, n_components: int = 0, pca_list=None, num_workers: int = 4):
    """PCA-based embeddings dimension reduction function.

    Split all array on parts by columns. Reduce each part with PCA. Concatenate results.

    Parameters:
    -------
    input_array (np.array): Input array.
    sub_dim (int): Dimension of each parts.
    n_components (int): Dimension of each parts after reduction. Used only if input parameter `pca_list` is not None.
    pca_list (np.array): list of PCA objects, to be applied to input_array
    num_workers (int): Amount of workers in Pool

    Returns
    -------
    out_array (np.array of floats): reduced input_array
    pca_list (list):
        List of PCA objects, fitted on input_array. Returned only if input parameter `pca_list` is None.
    """

    n_parts = (input_array.shape[1] - 1) // sub_dim + 1
    parts = [input_array[:, i*sub_dim: (i+1)*sub_dim] for i in range(n_parts)]
    if pca_list is None:
        args_list = [(part, n_components, None) for part in parts]
        return_pca = True
    else:
        args_list = [(part, n_components, pca) for part, pca in zip(parts, pca_list)]
        return_pca = False

    with Pool(num_workers) as p:
        transformed = p.map(reduce_part_, args_list)
    if pca_list is None:
        transformed, pca_list = zip(*transformed)
    transformed = np.concatenate(transformed, axis=1)
    
    if return_pca:
        return transformed, pca_list
    else:
        return transformed
