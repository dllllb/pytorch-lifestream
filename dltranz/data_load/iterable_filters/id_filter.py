from torch.utils.data.dataset import IterableDataset


class IdFilter(IterableDataset):
    def __init__(self, id_col, relevant_ids):
        self._id_col = id_col
        self._relevant_ids = set(relevant_ids)
        one_element = next(iter(self._relevant_ids))
        self._id_type = type(one_element)

        self._src = None

    def __call__(self, src):
        self._src = src
        return self

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
