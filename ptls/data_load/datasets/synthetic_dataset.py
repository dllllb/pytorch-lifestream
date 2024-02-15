import logging
import os
import warnings
from glob import glob
from itertools import chain
from typing import Union, List

import numpy as np
import torch
from omegaconf import ListConfig

from ptls.data_load import read_pyarrow_file
from ptls.data_load import IterableChain

logger = logging.getLogger(__name__)








class HMM:
    def __init__(self, states, hidden_states,
                 state_transition_tensors,
                 hidden_state_transition_matrix,
                 noise_hidden_state_transition_matrix=None, noise=0.):
        self.states = states
        self.n_states = len(self.states)
        self.hidden_states = hidden_states
        self.n_hidden_states = len(self.hidden_states)
        self.state_transition_tensors = state_transition_tensors
        self.hidden_state_transition_matrix = hidden_state_transition_matrix
        self.noise_hidden_state_transition_matrix = noise_hidden_state_transition_matrix
        self.noise = noise

        assert self.state_transition_tensors[0].ndim == 3
        assert self.state_transition_tensors[0].shape[0] == self.state_transition_tensors[0].shape[1] == self.n_states
        assert self.check_states_for_inds(self.states)
        assert self.check_states_for_inds(self.hidden_states)

        if type(self.hidden_state_transition_matrix) is list:
            assert len(self.hidden_state_transition_matrix) == len(self.state_transition_tensors)
            hidden_state_transition_matrix = self.hidden_state_transition_matrix[0]
        else:
            hidden_state_transition_matrix = self.hidden_state_transition_matrix

        assert hidden_state_transition_matrix.ndim == 2
        assert self.state_transition_tensors[0].shape[2] == hidden_state_transition_matrix.shape[0]
        assert hidden_state_transition_matrix.shape[0] == \
               hidden_state_transition_matrix.shape[1] == \
               self.n_hidden_states

        self.state = None
        self.h_state = None
        self.noise_h_state = None
        self.chosen_hidden_state_transition_matrix = None
        self.chosen_noise_hidden_state_transition_matrix = None
        self.chosen_transition_tensor = None
        self.reset()

    def gen_next(self):
        self.h_state = np.random.choice(self.hidden_states,
                                        p=self.chosen_hidden_state_transition_matrix[self.h_state.ind])
        self.state = np.random.choice(self.states,
                                      p=self.chosen_transition_tensor[self.state.ind, :, self.h_state.ind])

        if self.noise_hidden_state_transition_matrix is not None:
            self.noise_h_state = np.random.choice(self.hidden_states,
                                                  p=self.chosen_noise_hidden_state_transition_matrix[self.noise_h_state.ind])
            h_state = self.h_state if np.random.rand() >= self.noise else self.noise_h_state
        else:
            h_state = self.h_state if np.random.rand() >= self.noise else np.random.choice(self.hidden_states)
        return h_state, self.state

    def gen_seq(self, seq_len):
        a, b = list(), list()
        hs, s = self.reset()
        a.append(hs.unwrap())
        b.append(s.unwrap())

        for i in range(seq_len - 1):
            hs, s = self.gen_next()
            a.append(hs.unwrap())
            b.append(s.unwrap())

        a = {k: torch.Tensor([x[k] for x in a]) for k in a[0]}
        b = {k: torch.Tensor([x[k] for x in b]) for k in b[0]}
        return {**a, **b, 'event_time': torch.arange(seq_len)}

    def reset(self):
        self.h_state = np.random.choice(self.hidden_states)
        self.noise_h_state = np.random.choice(self.hidden_states)
        self.state = np.random.choice(self.states)
        return self.h_state, self.state

    def __len__(self):
        return len(self.state_transition_tensors)

    def gen_by_ind(self, item, seq_len):
        if type(self.hidden_state_transition_matrix) is list:
            self.chosen_hidden_state_transition_matrix = self.hidden_state_transition_matrix[item]
            self.chosen_noise_hidden_state_transition_matrix = self.noise_hidden_state_transition_matrix[item]
        else:
            self.chosen_hidden_state_transition_matrix = self.hidden_state_transition_matrix
            self.chosen_noise_hidden_state_transition_matrix = self.noise_hidden_state_transition_matrix
        self.chosen_transition_tensor = self.state_transition_tensors[item]
        return self.gen_seq(seq_len)

    @staticmethod
    def check_states_for_inds(states):
        return sorted([s.ind for s in states]) == [i for i in range(len(states))]


class IterableHMM(HMM):
    def __init__(self, states, hidden_states, transition_tensor_generator,
                 hidden_state_transition_matrix, noise=0., l=10000):
        test_tensor = next(transition_tensor_generator)
        super().__init__(states, hidden_states, test_tensor,
                 hidden_state_transition_matrix, noise)
        self.transition_tensor_generator = transition_tensor_generator
        self.l = l

    def gen_by_ind(self, item, seq_len):
        self.chosen_transition_tensor = next(self.transition_tensor_generator)
        return self.gen_seq(seq_len)

    def __len__(self):
        return self.l


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self,
                 hmms,
                 seq_len=1000,
                 post_processing=None,
                 i_filters: List = None):
        self.hmms = hmms
        self.seq_len = seq_len
        if i_filters is not None:
            self.post_processing = IterableChain(*i_filters)
        else:
            self.post_processing = post_processing
        if post_processing is not None:
            warnings.warn('`post_processing` parameter is deprecated, use `i_filters`')

    def __getitem__(self, item):
        c = item % len(self.hmms)
        ind = item // len(self.hmms)
        hmm = self.hmms[c]
        item = hmm.gen_by_ind(ind, self.seq_len)
        item['class_label'] = c
        if self.post_processing is not None:
            item = self.post_processing(item)
        return item

    def __len__(self):
        return sum([len(x) for x in self.hmms])

    @staticmethod
    def to_torch(x):
        if type(x) is np.ndarray and x.dtype.kind in ('i', 'f'):
            return torch.from_numpy(x)
        return x
