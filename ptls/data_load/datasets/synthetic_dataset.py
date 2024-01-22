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


class SquareSampler:
    def __init__(self, side, n_states, c=None):
        self.side = side
        self.n_states = n_states
        self.dim = self.n_states ** 2
        self.c = np.zeros((1, self.dim)) if c is None else c
        assert self.c.shape[-1] == self.dim

    def sample(self, n=1, to_matrix=True):
        x = ((np.random.rand(self.dim * n).reshape(-1, self.dim) - 0.5) * self.side) + self.c
        if to_matrix:
            x = x.reshape(n, self.n_states, self.n_states)
        return x


class SphereSampler:
    def __init__(self, n_states, c=None):
        self.n_states = n_states
        self.dim = self.n_states ** 2

    def sample(self, n=1, to_matrix=True):
        x = np.random.randn(n * self.dim).reshape(n, self.dim)
        if to_matrix:
            x = x.reshape(n, self.n_states, self.n_states)
        return x


class PlaneClassAssigner:
    def __init__(self, n_states, n_hidden_states, v=None, delta_shift=None):
        self.ready = False
        self.n_states = n_states
        self.n_hidden_states = n_hidden_states
        self.dim = self.n_states**2
        self.delta_shift = delta_shift
        self.v = v
        if self.v is not None:
            self.norm_vector()
            self.ready = True

    def set_random_vector(self):
        self.v = list()
        for i in range(self.n_hidden_states):
            v = np.random.randn(self.dim).reshape(1, -1)
            v = self.norm_vector(v)
            self.v.append(v)
        self.ready = True

    @staticmethod
    def norm_vector(v):
        return v / np.sqrt((v ** 2).sum())

    def get_class(self, x, h_state_n):
        v = self.v[h_state_n]
        c = np.where((v * x).sum(axis=-1) > 0, 1, -1)
        if self.delta_shift is not None:
            x = x + v * self.delta_shift * c.reshape(-1, 1)
        c = np.where(c == 1, 1, 0)
        return x, c


class TransitionTensorGenerator:
    def __init__(self, sampler, assigner, n_hidden_states):
        self.sampler = sampler
        self.assigner = assigner
        self.n_hidden_states = n_hidden_states

    def gen_tensors(self, n, soft_norm=True, sphere_norm=False, sphere_r=1):
        pos_tensors, neg_tensors = list(), list()
        for h in range(self.n_hidden_states):
            if not self.assigner.ready:
                self.assigner.set_random_vector()
            pos_matrices, neg_matrices = list(), list()
            n_pos_matrices, n_neg_matrices = 0, 0
            while n_pos_matrices < n or n_neg_matrices < n:
                raw_vectors = self.sampler.sample(2 * n - 2 * min(n_pos_matrices, n_neg_matrices), to_matrix=False)

                if sphere_norm:
                    raw_vectors = raw_vectors / np.sqrt((raw_vectors**2).sum(axis=-1, keepdims=True)) * sphere_r

                x, c = self.assigner.get_class(raw_vectors, h)
                x = x.reshape(x.shape[0], self.sampler.n_states, self.sampler.n_states)

                if soft_norm:
                    x = np.exp(x)/np.exp(x).sum(axis=-1, keepdims=True)

                pos_matrices.append(x[c == 1])
                neg_matrices.append(x[c == 0])
                n_pos_matrices += pos_matrices[-1].shape[0]
                n_neg_matrices += neg_matrices[-1].shape[0]
            pos_matrices = np.concatenate(pos_matrices, axis=0)[:n]
            neg_matrices = np.concatenate(neg_matrices, axis=0)[:n]
            pos_tensors.append(pos_matrices)
            neg_tensors.append(neg_matrices)
        pos_tensors = np.stack(pos_tensors, axis=-1)
        neg_tensors = np.stack(neg_tensors, axis=-1)
        return pos_tensors, neg_tensors


class Dist:
    def sample(self):
        raise NotImplemented


class BetaDist(Dist):
    def __init__(self, a, b, **kwargs):
        self.a = a
        self.b = b

    def sample(self):
        return np.random.beta(self.a, self.b)


class UniformDist(Dist):
    def __init__(self, **kwargs):
        pass

    def sample(self):
        return np.random.rand()


class ConstDist(Dist):
    def __init__(self, n, p, **kwargs):
        assert n == len(p)
        self.n = n
        self.p = p

    def sample(self):
        return np.random.choice(self.n, p=self.p) / self.n


class Feature:
    def sample(self):
        raise NotImplemented


class ConstFeature(Feature):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value


class CategoryFeature(Feature):
    def __init__(self, n, dist_type='uniform', dist_args={}):
        assert dist_type in ['uniform', 'beta', 'const']
        self.n = n

        if dist_type == 'uniform':
            self.dist = UniformDist()
        elif dist_type == 'beta':
            self.dist = BetaDist(**dist_args)
        elif dist_type == 'const':
            self.dist = ConstDist(n=n, **dist_args)

    def sample(self):
        return int(self.dist.sample() * self.n)


class FloatFeature(Feature):
    def __init__(self, min=0, max=1, log=False, dist_type='uniform', dist_args={}):
        assert dist_type in ['uniform', 'beta']
        self.min = min
        self.max = max
        self.log = log

        if dist_type == 'uniform':
            self.dist = UniformDist()
        elif dist_type == 'beta':
            self.dist = BetaDist(**dist_args)

    def sample(self):
        v = self.min + self.dist.sample() * (self.max - self.min)
        v = np.exp(v) if self.log else v
        return v


class State:
    def __init__(self, features, ind):
        self.features = features
        self.names = set(self.features.keys())
        self.ind = ind

    def unwrap(self):
        return {k: v.sample() for k, v in self.features.items()}


class HMM:
    def __init__(self, states, hidden_states, state_transition_tensors,
                 hidden_state_transition_matrix, noise=0.):
        self.states = states
        self.n_states = len(self.states)
        self.hidden_states = hidden_states
        self.n_hidden_states = len(self.hidden_states)
        self.state_transition_tensors = state_transition_tensors
        self.hidden_state_transition_matrix = hidden_state_transition_matrix
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
        self.chosen_hidden_state_transition_matrix = None
        self.chosen_transition_tensor = None
        self.reset()

    def gen_next(self):
        self.h_state = np.random.choice(self.hidden_states,
                                        p=self.chosen_hidden_state_transition_matrix[self.h_state.ind])
        self.state = np.random.choice(self.states,
                                      p=self.chosen_transition_tensor[self.state.ind, :, self.h_state.ind])
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
        self.state = np.random.choice(self.states)
        return self.h_state, self.state

    def __len__(self):
        return len(self.state_transition_tensors)

    def gen_by_ind(self, item, seq_len):
        if type(self.hidden_state_transition_matrix) is list:
            self.chosen_hidden_state_transition_matrix = self.hidden_state_transition_matrix[item]
        else:
            self.chosen_hidden_state_transition_matrix = self.hidden_state_transition_matrix
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
