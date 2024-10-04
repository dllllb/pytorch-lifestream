import torch
import numpy as np
from torch import nn


class HardNegativeNSelector:
    def __init__(self, neg_count=5):
        self.neg_count = neg_count
        
    def get_pairs(self, chains_vectors, slice_vectors, pos_slice_count):
        chains_count = chains_vectors.shape[0]
        slices_count = slice_vectors.shape[0]
        neg_chains, neg_slices = [], []
        
        def neg_sampling(i):
            nonlocal chains_vectors, slice_vectors, pos_slice_count
            nonlocal chains_count, slices_count
            nonlocal neg_chains, neg_slices
            neg_slices.append(slice_vectors[np.arange(slices_count) // pos_slice_count != i])
            neg_chains.append(chains_vectors[i:i+1].repeat_interleave(slices_count - pos_slice_count, dim=0))
            
        neg_sampling = [neg_sampling(i) for i in range(chains_count)]
        
        neg_chains = torch.cat(neg_chains, dim=0)
        neg_slices = torch.cat(neg_slices, dim=0)

        negative_loss = nn.functional.relu(1.0 - torch.sqrt(torch.sum((neg_chains - neg_slices) ** 2, dim=-1))).pow(2)
        negative_loss = torch.reshape(negative_loss, (chains_count, slices_count - pos_slice_count))
        idx = torch.topk(negative_loss, self.neg_count, dim=1).indices
        
        selected_neg_chains, selected_neg_slices = [], []
        
        def select(i):
            nonlocal neg_chains, neg_slices, idx, self
            nonlocal selected_neg_chains, selected_neg_slices
            selected_neg_chains.append(chains_vectors[i:i+1].repeat_interleave(self.neg_count, dim=0))
            selected_neg_slices.append(neg_slices[i*(slices_count - pos_slice_count):(i+1)*(slices_count - pos_slice_count)][idx[i]])
            
        selecting = [select(i) for i in range(chains_count)]
        
        selected_neg_chains = torch.cat(selected_neg_chains, dim=0)
        selected_neg_slices = torch.cat(selected_neg_slices, dim=0)
        
        pos_chains = chains_vectors.repeat_interleave(pos_slice_count, dim=0)
        pos_slices = slice_vectors
        
        return pos_chains, pos_slices, selected_neg_chains, selected_neg_slices
    