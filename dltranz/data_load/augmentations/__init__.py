from torch.utils.data.dataset import IterableDataset


class IterableAugmentations(IterableDataset):
    def __init__(self, *i_filters):
        self.i_filters = i_filters

        self._src = None

    def __call__(self, src):
        self._src = src

    def __iter__(self):
        for x, *targets in self._src:
            for f in self.i_filters:
                x = f(x)
            yield x, *targets
