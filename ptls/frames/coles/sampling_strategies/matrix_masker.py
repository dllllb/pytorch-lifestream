import torch


class MatrixMasker:
    """
    Returns matrix masked with zeros for
    summing only positive pairs distances
    (e.g. (2k-1, 2k) and (2k, 2k - 1) for k in 1 to N).
    """
    def get_masked_matrix(self, matrix, classes):
        mask = (classes.unsqueeze(1) == classes.unsqueeze(0)).int()
        mask.fill_diagonal_(0)
        masked_matrix = matrix*mask
        return masked_matrix
