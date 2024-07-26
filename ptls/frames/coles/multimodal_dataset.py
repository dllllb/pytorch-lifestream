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
            column name with name sources, must be specified in the same order as trx_encoders in 
            ptls.frames.coles.multimodal_module.MultiModalSortTimeSeqEncoderContainer
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

        local_times_tensor = torch.tensor([x[0] for x in common_local_time])
        indexes = self.splitter.split(local_times_tensor)

        res_ind = []
        for inds in indexes:
            dct = defaultdict(list)
            for ind in inds:
                _, loc_index, src_name = common_local_time[ind]
                dct[src_name].append(loc_index)
            res_ind.append(dct)

        for src_name, feature_dict in feature_arrays.items():
            if src_name != self.col_id:
                filtered_features = {k: v for k, v in feature_dict.items() if self.is_seq_feature(k, v)}
                res[src_name] = [{k: v[ix[src_name]] for k, v in filtered_features.items()} for ix in res_ind]

        return res

    def collate_fn(self, batch, return_dct_labels=False):
        dict_class_labels = get_dict_class_labels(batch)
        batch = reduce(lambda x, y: {k: x[k] + y[k] for k in x if k in y}, batch)
        
        padded_batch = collate_multimodal_feature_dict(batch)
        source_indices = {source: index for index, source in enumerate(self.source_names)}
        
        # common_local_time is a tensor containing information about event indexes from each modality for each sample
        # dim_0 - the number of samples in the batch
        # dim_1 - total length of the sample (including all modalities)
        # dim_2 = 3 - event_time, index, source
        dim_0 = padded_batch[self.source_names[0]].payload[self.col_time].shape[0]
        dim_1 = sum(subseq.payload[self.col_time].shape[1] for source_name, subseq in padded_batch.items())
        common_local_time_shape = (dim_0, dim_1, 3)

        common_local_time = torch.zeros(common_local_time_shape, dtype=torch.double)

        current_dim_1 = 0
        for source_name, subseq in padded_batch.items():
            source_index = source_indices[source_name]
            
            # current_dim_1_end - a pointer to the index from which to fill in the next modality
            current_dim_1_end = current_dim_1 + subseq.payload[self.col_time].shape[1]
            
            # each modality within itself is already sorted by time 
            # so you can take arange as indexes inside the modality
            common_local_time[:, current_dim_1:current_dim_1_end, 1] = torch.arange(subseq.payload[self.col_time].shape[1])
            common_local_time[:, current_dim_1:current_dim_1_end, 2] = source_index
            
            # subseq_local_times - event_time for each modality, padding is replaced by inf
            subseq_local_times = subseq.payload[self.col_time].type(torch.FloatTensor)
            subseq_local_times_masked = subseq_local_times.masked_fill_(subseq_local_times == 0, float('inf'))
            common_local_time[:, current_dim_1:current_dim_1_end, 0] = subseq_local_times_masked
            current_dim_1 = current_dim_1_end
        
        # getting indexes sorted by event_time order
        indices = common_local_time[:,:,0].sort(dim=1, stable=True)[1]
        
        # gathering of the final tensor containing the order of the indies within each modality
        # and a pointer to the modality to which the index belongs
        common_local_time_sorted = torch.gather(common_local_time, 1, indices.unsqueeze(-1).expand(-1, -1, 3))[:, :, 1:]      
        if return_dct_labels:
            return (padded_batch, common_local_time_sorted.short()), dict_class_labels
        return (padded_batch, common_local_time_sorted.short()), dict_class_labels[list(dict_class_labels.keys())[0]]

    
class MultiModalIterableDataset(MultiModalDataset, torch.utils.data.IterableDataset):
    pass