from torch.utils.data.dataset import IterableDataset


class TargetExtractor(IterableDataset):
    def __init__(self, target_col):
        """Extract value from `target_col` and mention it as `y`

        for x, * in seq:
            y = x[target_col]
            yield x, y

        Args:
            target_col: field where `y` is stored

        """
        self._target_col = target_col

        self._src = None

    def __call__(self, src):
        self._src = src
        return self

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            y = features[self._target_col]
            yield rec, y
