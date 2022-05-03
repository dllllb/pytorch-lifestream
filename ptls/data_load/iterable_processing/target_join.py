from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class TargetJoin(IterableProcessingDataset):
    def __init__(self, id_col, target_values, func=int):
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

        self._id_col = id_col
        self._target_values = target_values
        self.func = func

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            _id = features[self._id_col]
            y = self.func(self._target_values[_id])
            yield features, y
