import pandas as pd
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
from ptls.data_load.augmentations.seq_len_limit import SeqLenLimit
import numpy as np
import torch


class Filtering(IterableProcessingDataset):
    def __init__(self, mode: str, 
                 id_col: str = None, 
                 relevant_ids: list = None,
                 category_max_size: dict = None,
                 replace_value: str = 'max',
                 min_seq_len: int = None, 
                 max_seq_len: int = None, 
                 seq_len_col: str = None,
                 sequence_col: str = None,
                 df_relevant_ids: pd.DataFrame=None,
                 strategy: str = 'tail'):
        super().__init__()
        self.mode = mode
        self._id_col = id_col
        self._relevant_ids = set(relevant_ids) if relevant_ids is not None else None
        self._id_type = type(next(iter(relevant_ids))) if relevant_ids is not None else None
        self._category_max_size = category_max_size
        self._replace_value = replace_value
        self._min_seq_len = min_seq_len
        self._max_seq_len = max_seq_len
        self._seq_len_col = seq_len_col
        self._sequence_col = sequence_col
        self._df_relevant_ids = set(tuple(r) for r in df_relevant_ids.values.tolist()) if df_relevant_ids is not None else None
        self.id_columns = df_relevant_ids.columns if df_relevant_ids is not None else None
        self.id_types = [type(v) for v in next(iter(self._df_relevant_ids))] if df_relevant_ids is not None else None
        self.proc = SeqLenLimit(max_seq_len, strategy) if max_seq_len is not None else None

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if isinstance(rec, tuple) else rec

            if self.mode == 'DeleteNan':
                for name, value in features.items():
                    if value is None:
                        features[name] = torch.Tensor([])
                yield rec

            elif self.mode == 'IdFilter':
                # Required to have id_col and relevant_ids
                _id = features[self._id_col]
                if not self._is_in_relevant_ids_with_type(_id):
                    continue
                yield rec

            elif self.mode == 'CategorySizeClip':
                # Required to have category_max_size
                for name, max_size in self._category_max_size.items():
                    features[name] = self._smart_clip(features[name], max_size)
                yield rec

            elif self.mode == 'SeqLenFilter':
                seq_len = self.get_len(features)
                if self._min_seq_len is not None and seq_len < self._min_seq_len:
                    continue
                if self._max_seq_len is not None and seq_len > self._max_seq_len:
                    continue
                yield rec

            elif self.mode == 'ISeqLenLimit':
                features = self.proc(features)
                yield features

            elif self.mode == 'FilterNonArray':
                to_del = [k for k, v in features.items() if not isinstance(v, (np.ndarray, torch.Tensor))]
                for k in to_del:
                    del features[k]
                yield rec

            elif self.mode == 'IdFilterDf':
                _id = tuple([col_type(features[col]) for col, col_type in zip(self.id_columns, self.id_types)])
                if _id not in self._df_relevant_ids:
                    continue
                yield rec

            else:
                raise ValueError("Unsupported mode")

    def _is_in_relevant_ids_with_type(self, _id):
        if type(_id) is not self._id_type:
            raise TypeError(f'Type mismatch when id check. {type(_id)} found in sequence, '
                            f'but {self._id_type} from relevant_ids expected')
        return _id in self._relevant_ids

    def _smart_clip(self, values, max_size):
        if self._replace_value == 'max':
            return values.clip(0, max_size - 1)
        else:
            return torch.from_numpy(np.where((0 <= values) & (values < max_size), values, self._replace_value))

    def get_len(self, rec):
        if self._seq_len_col is not None:
            return rec[self._seq_len_col]
        return len(rec[self.get_sequence_col(rec)])
