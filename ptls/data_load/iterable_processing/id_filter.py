from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class IdFilter(IterableProcessingDataset):
    def __init__(self, id_col, relevant_ids):
        """Remove records which are not in `relevant_ids`

        Args:
            id_col: field where id is stored
            relevant_ids: list of ids which should be kept
        """
        super().__init__()

        self._id_col = id_col
        self._relevant_ids = set(relevant_ids)
        one_element = next(iter(self._relevant_ids))
        self._id_type = type(one_element)

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            _id = features[self._id_col]
            if not self._is_in_relevant_ids_with_type(_id):
                continue
            yield rec

    def _is_in_relevant_ids_with_type(self, _id):
        if type(_id) is not self._id_type:
            raise TypeError(f'Type mismatch when id check. {type(_id)} found in sequence, '
                            f'but {self._id_type} from relevant_ids expected')

        return _id in self._relevant_ids
