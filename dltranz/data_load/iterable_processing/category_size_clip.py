from torch.utils.data.dataset import IterableDataset
import numpy as np


class CategorySizeClip(IterableDataset):
    def __init__(self, category_max_size, replace_value='max'):
        """

        Args:
            category_max_size: {field_name, max_size}
            replace_value: value for infrequent categories, int for specific value, 'max' for `category_max_size - 1`
        """
        self._category_max_size = category_max_size
        self._replace_value = replace_value

        self._src = None

    def __call__(self, src):
        self._src = src
        return self

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec

            # clip embeddings dictionary by max value
            for name, max_size in self._category_max_size.items():
                features[name] = self._smart_clip(features[name], max_size)

            yield rec

    def _smart_clip(self, values, max_size):
        if self._replace_value == 'max':
            return values.clip(0, max_size - 1)
        else:
            return np.where((0 <= values) & (values < max_size), values, self._replace_value)
