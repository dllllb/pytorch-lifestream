import numpy as np
import torch

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class CategorySizeClip(IterableProcessingDataset):
    def __init__(self, category_max_size, replace_value='max'):
        """

        Args:
            category_max_size: {field_name, max_size}
            replace_value: value for infrequent categories, int for specific value, 'max' for `category_max_size - 1`
        """
        super().__init__()

        self._category_max_size = category_max_size
        self._replace_value = replace_value

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
            return torch.from_numpy(np.where((0 <= values) & (values < max_size), values, self._replace_value))
