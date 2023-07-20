from functools import reduce
from operator import iadd
import numpy as np

import torch

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles.split_strategy import AbsSplit



class ColesDatasetTime(FeatureDict, torch.utils.data.Dataset):
    """Dataset for ptls.frames.coles.CoLESModule
    Parameters
    ----------
    data:
        source data with feature dicts
    splitter:
        object from from `ptls.frames.coles.split_strategy`.
        Used to split original sequence into subsequences which are samples from one client.
    col_time:
        column name with event_time
    time_margin: 
        time margin for choice negative and positive pairs.
    """

    def __init__(self,
                 data,
                 splitter: AbsSplit,
                 time_margin=100,
                 col_time='event_time',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.data = data
        self.splitter = splitter
        self.col_time = col_time
        self.time_margin = time_margin

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        return self.get_splits(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.get_splits(feature_arrays)

    def get_splits(self, feature_arrays):
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        return [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} for ix in indexes]
    
    def get_res_class_labels(self, mask_matrix):
        n = mask_matrix.shape[0]
        class_labels = np.arange(n, 2*n)
        
        x = (mask_matrix == 0).int()
        ind_pos_row, ind_pos_col = x.nonzero(as_tuple=True)
        
        class_labels[ind_pos_col] = ind_pos_row
        
        return class_labels
    
    def get_raw_class_labels(self, batch):
        class_labels = [] 
        event_time_max = [] 
        event_time_min = [] 
        for i, class_samples in enumerate(batch):
            for sample in class_samples:
                max_time = max(sample[self.col_time])
                min_time = min(sample[self.col_time])
                
                event_time_max.append(max_time)
                event_time_min.append(min_time)
                class_labels.append(i)
        
        return class_labels, event_time_max, event_time_min
              
    def get_labels_by_diff_time(self, batch):
        # Получим отношение сплита к юзеру без учета времени, а также максимальное и минимальное время кадого сплита
        class_labels, event_time_max, event_time_min = self.get_raw_class_labels(batch)
        
        tensor_labels = torch.tensor(class_labels)
        tensor_max_time = torch.tensor(event_time_max)
        tensor_min_time = torch.tensor(event_time_min)
        
        n = tensor_labels.size(0)
        
        # Матрица разностей макисмального и минимального времени для юзеров
        diff_time_matrix = (tensor_min_time.expand(n, n) - tensor_max_time.expand(n, n).t()).clip(0, None)
        # Сравниваем с отсупом
        x_times = (diff_time_matrix >= self.time_margin).int()
        # Делаем симметричной
        x_times_res = torch.max(x_times, x_times.t())
        
        lbl_diff = tensor_labels.expand(n, n) - tensor_labels.expand(n, n).t()
        x = (lbl_diff != 0).int()
        
        x_res = torch.max(x_times_res, x)# здесь получили позитивные пары
        # теперь соберем классы. Будем двигаться по массиву двумя указателями
        class_labels = self.get_res_class_labels(x_res)
                
        return class_labels
          
    def collate_fn(self, batch):
        class_labels = self.get_labels_by_diff_time(batch)
        batch = reduce(iadd, batch)
        padded_batch = collate_feature_dict(batch)
        return padded_batch, torch.LongTensor(class_labels)

    
class ColesIterableDatasetTime(ColesDatasetTime, torch.utils.data.IterableDataset):
    pass