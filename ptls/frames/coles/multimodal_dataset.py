import pandas as pd
import numpy as np
import torch
from functools import partial
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.data_load.datasets import inference_data_loader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import logging
import pickle
from itertools import groupby
from functools import reduce
from operator import iadd
from collections import defaultdict
from ptls.data_load.feature_dict import FeatureDict
from ptls.frames.coles.split_strategy import AbsSplit
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


def collate_feature_dict(batch):
    """Collate feature with arrays to padded batch

    Check feature consistency. Keys for all batch samples should be the same.
    Convert scalar value to tensors like target col

    Parameters
    ----------
    batch:
        list with feature dicts
    Returns
    -------
        PaddedBatch
    """
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
            else:
                new_x[k] = v
    return PaddedBatch(new_x, lengths)


def collate_target(x, num=1):
    vec = np.array(x, dtype=np.float32)
    if num == 1:
        return vec.sum()
    elif abs(num) >= len(vec):
        return vec
    elif num < 0:
        return vec[:abs(num)]
    else:
        return np.hstack((vec[:num-1], vec[num-1:].sum()))[:len(vec)]
    
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
        col_id='epk_id',
        col_time='event_time',
        source_names=('way4', 'cod'),
        *args, **kwargs
    ):
        """
        Dataset for multimodal learning.
        Parameters:
        -----------
        datas:
            tuple of source data with feature dicts.
        splitters:
            tuple of objects from from `ptls.frames.coles.split_strategy`.
            Used to split original sequence into subsequences which are samples from one client.
        col_time:
            column name with event_time
        col_id:
            column name with user_id
        col_name_source:
            column name with name sources
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
        return self.get_splits(feature_arrays)
    
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



class MultiModalSortTimeSeqEncoderContainer(torch.nn.Module):
    def __init__(self,
                 trx_encoders,
                 seq_encoder_cls, 
                 input_size,
                 is_reduce_sequence=True,
                 col_time='event_time',
                 **seq_encoder_params
                ):
        super().__init__()
        
        self.trx_encoders = torch.nn.ModuleDict(trx_encoders)
        self.seq_encoder = seq_encoder_cls(
            input_size=input_size,
            is_reduce_sequence=is_reduce_sequence,
            **seq_encoder_params,
        )
        
        self.col_time = col_time
    
    @property
    def is_reduce_sequence(self):
        return self.seq_encoder.is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self.seq_encoder.is_reduce_sequence = value

    @property
    def embedding_size(self):
        return self.seq_encoder.embedding_size
    
    def get_tensor_by_indices(self, tensor, indices):
        batch_size = tensor.shape[0]
        return tensor[:, indices, :][torch.arange(batch_size), torch.arange(batch_size), :, :]
        
    def merge_by_time(self, x):
        device = list(x.values())[1][0].device
        batch, batch_time = torch.tensor([], device=device), torch.tensor([], device=device)
        for source_batch in x.values():
            if source_batch[0] != 'None':
                batch = torch.cat((batch, source_batch[1].payload), dim=1)
                batch_time = torch.cat((batch_time, source_batch[0]), dim=1)
        
        batch_time[batch_time == 0] = float('inf')
        indices_time = torch.argsort(batch_time, dim=1)
        batch = self.get_tensor_by_indices(batch, indices_time)
        return batch
            
    def trx_encoder_wrapper(self, x_source, trx_encoder, col_time):
        if torch.nonzero(x_source.seq_lens).size()[0] == 0:
            return x_source.seq_lens, 'None', 'None'
        return x_source.seq_lens, x_source.payload[col_time], trx_encoder(x_source)
        
    def multimodal_trx_encoder(self, x):
        res = {}
        tmp_el = list(x.values())[0]
        
        batch_size = tmp_el.payload[self.col_time].shape[0]
        length = torch.zeros(batch_size, device=tmp_el.device).int()
        
        for source, trx_encoder in self.trx_encoders.items():
            enc_res = self.trx_encoder_wrapper(x[source], trx_encoder, self.col_time)
            source_length, res[source] = enc_res[0], (enc_res[1], enc_res[2])
            length = length + source_length
        return res, length
            
    def forward(self, x):
        x, length = self.multimodal_trx_encoder(x)
        x = self.merge_by_time(x)
        padded_x = PaddedBatch(payload=x, length=length)
        x = self.seq_encoder(padded_x)
        return x
    
    
class MultiModalInferenceDataset(FeatureDict, torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        source_features,
        source_names=('way4', 'cod'),
        col_id='epk_id',
        col_time='event_time',
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.data = data
        self.col_time = col_time
        self.col_id = col_id
        self.source_names = source_names
        self.source_features = source_features
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        return self.get_splits(feature_arrays)
    
    def __iter__(self):
        for feature_arrays in self.data:
            split_data = self.split_source(feature_arrays)
            #print(split_data)
            yield split_data
            
    def split_source(self, feature_arrays):
        res = defaultdict(dict)
        for feature_name, feature_array in feature_arrays.items():
            if feature_name == self.col_id:
                res[self.col_id] = feature_array
                #continue
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
    
    @staticmethod
    def collate_fn(batch, return_dct_labels=False, col_id = 'epk_id'):
        batch_ids = []
        for sample in batch:
            batch_ids.append(sample[col_id][0])
            del sample[col_id]
        dict_class_labels = get_dict_class_labels(batch)
        
        batch = reduce(lambda x, y: {k: x[k] + y[k] for k in x if k in y}, batch)
        padded_batch = collate_multimodal_feature_dict(batch)
        if return_dct_labels:
            return padded_batch, dict_class_labels
        return padded_batch, batch_ids

    
class MultiModalInferenceIterableDataset(MultiModalInferenceDataset, torch.utils.data.IterableDataset):
    pass

class MultiModalSupervisedDataset(FeatureDict, torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        source_features,
        source_names=('way4', 'cod'),
        col_id='epk_id',
        col_time='event_time',
        
        target_name = None,
        target_dtype = None,
        *args, **kwargs
    ):
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
        return feature_arrays
    
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