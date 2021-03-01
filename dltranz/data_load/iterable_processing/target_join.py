from dltranz.data_load.iterable_processing_dataset import IterableProcessingDataset
import numpy as np
import ast


class TargetJoin(IterableProcessingDataset):
    def __init__(self, id_col, target_values, distribution_targets_task):
        """Extract value from `target_values` by id_col and mention it as `y`

        for x, * in seq:
            id = x[id_col]
            y = target_values[id]
            yield x, y

        Args:
            id_col: field where `id` is stored
            target_values: dict with target

        """
        super().__init__()

        self.distribution_targets_task = distribution_targets_task
        self._id_col = id_col
        self._target_values = target_values

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            _id = features[self._id_col]

            if not self.distribution_targets_task:
                y = int(self._target_values[_id])
            else:
                y = self._target_values[_id]
                y = np.array(ast.literal_eval(y), dtype=object)
            yield features, y
