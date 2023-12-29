import numpy as np
import torch
from functools import reduce
from collections import defaultdict
from ptls.data_load.feature_dict import FeatureDict
from ptls.frames.coles.multimodal_dataset import collate_feature_dict, collate_multimodal_feature_dict, get_dict_class_labels
            

class MultiModalSupervisedDataset(FeatureDict, torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        source_features,
        source_names,
        col_id='client_id',
        col_time='event_time',
        
        target_name = None,
        target_dtype = None,
        *args, **kwargs
    ):
        """
        Dataset for multimodal supervised learning.
        Parameters:
        -----------
        data:
            concatinated data with feature dicts.
        source_features:
            list of column names 
        col_id:
            column name with user_id
        source_names:
            column name with name sources
        col_time:
            column name with event_time
        target_name:
            column name with target_name
        target_dtype:
            int or float
        """
        super().__init__(*args, **kwargs)
        
        self.data = data
        self.col_time = col_time
        self.col_id = col_id
        self.source_names = source_names
        self.source_features = source_features
        
        self.target_name = target_name
        self.target_dtype = target_dtype
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        return self.split_source(feature_arrays)
    
    def __iter__(self):
        for feature_arrays in self.data:
            split_data = self.split_source(feature_arrays)
            yield split_data
            
    def split_source(self, feature_arrays):
        res = defaultdict(dict)
        for feature_name, feature_array in feature_arrays.items():
            if feature_name == self.col_id:
                res[self.col_id] = feature_array
                #continue
            elif feature_name == self.target_name:
                res[self.target_name] = feature_array
            else:
                source_name, feature_name_transform = self.get_names(feature_name)
                res[source_name][feature_name_transform] = feature_array
        for source in self.source_names:
            if source not in res:
                res[source] = {source_feature: torch.tensor([]) for source_feature in self.source_features[source]}
        res1 = {}
        for source in res:
            res1[source] = [res[source]]
        return res1
    
    def get_names(self, feature_name):
        idx_del = feature_name.find('_')
        return feature_name[:idx_del], feature_name[idx_del + 1:]
            
    
    def collate_fn(self, batch, return_dct_labels=False):
        dict_class_labels = get_dict_class_labels(batch)
        batch_y = []
        for sample in batch:
            batch_y.append(sample[self.target_name][0])
            del sample[self.target_name]
        batch = reduce(lambda x, y: {k: x[k] + y[k] for k in x if k in y}, batch)
        padded_batch = collate_multimodal_feature_dict(batch)
        return padded_batch, torch.Tensor(batch_y)

    
class MultiModalSupervisedIterableDataset(MultiModalSupervisedDataset, torch.utils.data.IterableDataset):
    pass