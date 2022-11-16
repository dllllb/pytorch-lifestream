import torch


class MatrixMasker:
    """
    Returns matrix masked with zeros for
    summing only positive pairs distances
    (e.g. (2k-1, 2k) and (2k, 2k - 1) for k in 1 to N).
    """

    def __init__(self, split_count):
        self.split_count = split_count

    def get_masked_matrix(self, matrix, classes):
        num_classes = len(classes) // self.split_count
        mask = torch.zeros_like(matrix, dtype=torch.long)
        for i in range(num_classes):
            mask[i*self.split_count:(i + 1)*self.split_count, i*self.split_count:(i + 1)*self.split_count] = 1
        mask.fill_diagonal_(0)
        masked_matrix = matrix*mask
        return masked_matrix
