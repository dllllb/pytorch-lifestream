from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import AbsSplit
import numpy as np
from ptls.frames.coles.split_strategy import SampleSlices
from functools import reduce
from operator import iadd
import torch
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.feature_dict import FeatureDict


class InfoMaxDataset(FeatureDict, torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 splitter: AbsSplit,
                 col_time='event_time',
                 outside_split_count=5,  # count of outside splits on left and right outside sides
                 neg_cnt_min=25,
                 neg_cnt_max=200,
                 sample_chains=True,  # if true, we take full user chain for learning
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = data
        self.splitter = splitter
        self.col_time = col_time
        self.sample_chains = sample_chains
        self.outside_split_count = outside_split_count
        self.neg_cnt_min = neg_cnt_min
        self.neg_cnt_max = neg_cnt_max
    
    def get_splits(self, feature_arrays): # return pieces and piece boundaries
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        return [({k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)}, (ix[0], ix[-1])) for ix in indexes]
    
    def split_chain(self, feature_arrays):
        chain_len = len(feature_arrays[self.col_time])
        splitter = SampleSlices(5, chain_len - 200, chain_len - 100) # choose correctly size for chain
        ix = splitter.split(feature_arrays[self.col_time])
        return [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} for ix in ix]
    
    def one_split(self, feature_arrays, min_cnt, max_cnt):
        splitter = SampleSlices(1, min_cnt, max_cnt)
        local_date = feature_arrays[self.col_time]
        ix = splitter.split(local_date)
        return {k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)}
    
    def slice_negatives(self, feature_arrays, l, r):
        outside_ix = np.arange(l, r)
        outside_slice = {k: v[outside_ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)}
        local_date = outside_slice[self.col_time]
        if r - l > self.neg_cnt_max:
            cnt_max = self.neg_cnt_max
        else:
            cnt_max = r - l
        splitter = SampleSlices(self.outside_split_count, self.neg_cnt_min, cnt_max)
        indexes = splitter.split(local_date)
        return [{k: v[ix] for k, v in outside_slice.items() if self.is_seq_feature(k, v)} for ix in indexes]
    
    def get_negatives(self, feature_arrays):
        length = len(feature_arrays[self.col_time])
        inner_pos = np.random.randint(low=self.neg_cnt_min+5, high=length-(self.neg_cnt_min+5))
        left_slices = self.slice_negatives(feature_arrays, 0, inner_pos)
        right_slices = self.slice_negatives(feature_arrays, inner_pos + 1, length)
        negatives = (left_slices, right_slices)
        return negatives
        
    def get_inner_splits(self, feature_arrays):
        length = len(feature_arrays[self.col_time])
        split_items = self.get_splits(feature_arrays)
        splits = [split_item[0] for split_item in split_items]
        splits_lens = [split_item[1] for split_item in split_items]
        pos_inner_splits = [self.one_split(split, 5, 10) for split in splits]
        positives = (splits, pos_inner_splits)
        negatives = self.get_negatives(feature_arrays)
        return positives, negatives
    
    def sample_chains_splits(self, feature_arrays):
        chain_splits = self.split_chain(feature_arrays)
        splits = [self.one_split(chain, self.neg_cnt_min, self.neg_cnt_max) for chain in chain_splits]
        return chain_splits, splits
    
    def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        inner_splits = self.get_inner_splits(feature_arrays)
        if self.sample_chains:
            feature_arrays, splits = self.sample_chains_splits(feature_arrays)
        else:
            splits = self.get_splits(feature_arrays)
        return feature_arrays, splits, inner_splits
    
    def __iter__(self):
        for feature_arrays in self.data:
            inner_splits = self.get_inner_splits(feature_arrays)
            if self.sample_chains:
                feature_arrays, splits = self.sample_chains_splits(feature_arrays)
            else:
                splits = self.get_splits(feature_arrays)
            yield feature_arrays, splits, inner_splits

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        slice_count = len(batch[0][1])
        chains = [slice_pair[0] for slice_pair in batch]
        splits = [split[1] for split in batch]
        reduced_splits = reduce(iadd, splits)
        
        if type(chains[0]) == list:
            chains = reduce(iadd, chains)
        
        slices = [split for split in reduced_splits]
        inner_pos_splits = [split_ for cl in batch for split_ in cl[2][0][0]]
        inner_pos_slices = [slice_ for cl in batch for slice_ in cl[2][0][1]]
        inner_neg_splits = [split_ for cl in batch for split_ in cl[2][1][0]]
        inner_neg_slices = [slice_ for cl in batch for slice_ in cl[2][1][1]]
        slices = slices + inner_neg_splits + inner_neg_slices
        inner_pos_count = len(inner_pos_splits)
        all_slice_count = slice_count * len(batch)
        return collate_feature_dict(chains), collate_feature_dict(slices), None, slice_count, inner_pos_count, all_slice_count


class InfoMaxIterableDataset(InfoMaxDataset, torch.utils.data.IterableDataset):
    pass
