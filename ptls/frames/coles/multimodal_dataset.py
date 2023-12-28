import numpy as np
import torch
from functools import reduce
from collections import defaultdict
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.padded_batch import PaddedBatch
from ptls.frames.coles import MultiModalSortTimeSeqEncoderContainer


def collate_feature_dict(batch):
    new_x_ = defaultdict(list)
    for i, x in enumerate(batch):
        for k, v in x.items():
            new_x_[k].append(v)

    seq_col = next(k for k, v in batch[0].items() if FeatureDict.is_seq_feature(k, v))
    lengths = torch.LongTensor([len(rec[seq_col]) for rec in batch])
    new_x = {}
    for k, v in new_x_.items():
        if type(v[0]) is torch.Tensor:
            if k.startswith('target'):
                new_x[k] = torch.stack(v, dim=0)
            else:
                new_x[k] = torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
        elif type(v[0]) is np.ndarray:
            new_x[k] = v  # list of arrays[object]
        else:
            v = np.array(v)
            if v.dtype.kind == 'i':
                new_x[k] = torch.from_numpy(v).long()
            elif v.dtype.kind == 'f':
                new_x[k] = torch.from_numpy(v).float()
            elif v.dtype.kind == 'b':
                new_x[k] = torch.from_numpy(v).bool()
            else:
                new_x[k] = v
    return PaddedBatch(new_x, lengths)

    
def collate_multimodal_feature_dict(batch):
    res = {}
    for source, source_batch in batch.items():
        res[source] = collate_feature_dict(source_batch)
    return res
    
def get_dict_class_labels(batch):
    res = defaultdict(list)
    for i, samples in enumerate(batch):
        for source, values in samples.items():
            for _ in values:
                res[source].append(i)
    for source in res:
        res[source] = torch.LongTensor(res[source])
    return dict(res)
            

class MultiModalDataset(FeatureDict, torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        splitter,
        source_features,
        col_id,
        source_names,
        col_time='event_time',
        *args, **kwargs
    ):
        """
        Dataset for multimodal learning.
        Parameters:
        -----------
        data:
            concatinated data with feature dicts.
        splitter:
            object from from `ptls.frames.coles.split_strategy`.
            Used to split original sequence into subsequences which are samples from one client.
        source_features:
            list of column names 
        col_id:
            column name with user_id
        source_names:
            column name with name sources
        col_time:
            column name with event_time
        """
        super().__init__(*args, **kwargs)
        
        self.data = data
        self.splitter = splitter
        self.col_time = col_time
        self.col_id = col_id
        self.source_names = source_names
        self.source_features = source_features
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        split_data = self.split_source(feature_arrays)
        return self.get_splits(split_data)
    
    def __iter__(self):
        for feature_arrays in self.data:
            split_data = self.split_source(feature_arrays)
            yield self.get_splits(split_data)
            
    def split_source(self, feature_arrays):
        res = defaultdict(dict)
        for feature_name, feature_array in feature_arrays.items():
            if feature_name == self.col_id:
                res[self.col_id] = feature_array
            else:
                source_name, feature_name_transform = self.get_names(feature_name)
                res[source_name][feature_name_transform] = feature_array
        for source in self.source_names:
            if source not in res:
                res[source] = {source_feature: torch.tensor([]) for source_feature in self.source_features[source]}
        return res
    
    def get_names(self, feature_name):
        idx_del = feature_name.find('_')
        return feature_name[:idx_del], feature_name[idx_del + 1:]
                
    
    def get_splits(self, feature_arrays):
        res = {}
        common_local_time = []
        for source_name, feature_dict in feature_arrays.items():
            if source_name != self.col_id:
                local_date = feature_dict[self.col_time]
                common_local_time.extend([(int(loc), ind, source_name) for ind, loc in enumerate(local_date)])
        common_local_time.sort(key=lambda x: x[0])
       
        indexes = self.splitter.split(torch.tensor([x[0] for x in common_local_time]))
        res_ind = []
        for inds in indexes:
            dct = defaultdict(list)
            for ind in inds:
                dct[common_local_time[ind][2]].append(common_local_time[ind][1])
            res_ind.append(dct)  
                
        for source_name, feature_dict in feature_arrays.items():
            if source_name != self.col_id:
                res[source_name] = [{k: v[ix[source_name]] for k, v in feature_dict.items() if self.is_seq_feature(k, v)} for ix in res_ind]
        return res
        
    def collate_fn(self, batch, return_dct_labels=False):
        dict_class_labels = get_dict_class_labels(batch)
        batch = reduce(lambda x, y: {k: x[k] + y[k] for k in x if k in y}, batch)
        padded_batch = collate_multimodal_feature_dict(batch)
        if return_dct_labels:
            return padded_batch, dict_class_labels
        return padded_batch, dict_class_labels[list(dict_class_labels.keys())[0]]

    
class MultiModalIterableDataset(MultiModalDataset, torch.utils.data.IterableDataset):
    pass