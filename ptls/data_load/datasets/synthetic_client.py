import numpy as np
from collections import defaultdict
from copy import deepcopy
import pickle
from .synthetic_utils import correct_round


class SyntheticClient:
    def __init__(self, labels, config, schedule):
        self.labels = labels
        self.config = config
        self.schedule = schedule
        self.chains = self.config.get_chains()
        self.sampler = SphereSampler(self.config.sampling_conf, self.config.class_assigners)

        self.curr_states = None
        self.curr_h_states = None
        self.timestamp = 0

        self.norm_labels = dict()
        for y in self.labels:
            self.norm_labels[y] = 1 if self.labels[y] else -1

        self.set_transition_tensors()

    def set_transition_tensors(self):
        tensors_dict = self.sampler.sample(self.labels)
        noise_tensor_dict = self.sampler.sample(self.labels)
        for ch in tensors_dict:
            self.chains[ch].set_transition_tensor(tensors_dict[ch])
            if self.chains[ch].add_noise_tensor:
                self.chains[ch].set_noise_tensor(noise_tensor_dict[ch])

    def init_chains(self):
        self.curr_states = dict()
        self.curr_h_states = dict()
        for ch in self.chains:
            self.curr_states[ch] = self.chains[ch].reset()
            self.curr_h_states[ch] = self.curr_states[ch]

    def gen_seq(self, l=512):
        seq = defaultdict(list)
        self.init_chains()

        for i in range(l):
            next_chain_name = self.schedule.get_next_chain(self.timestamp)

            h_state_chains = list(self.config.conj_from[next_chain_name])
            if len(h_state_chains):
                h_state = 0
                skip = 1
                for ch in sorted(h_state_chains):
                    h_state += skip * self.curr_h_states[ch]
                    skip *= self.chains[ch].n_states
            else:
                h_state = None

            new_state, new_h_state = self.chains[next_chain_name].next_state(h_state)
            t2s = self.chains[next_chain_name].transition2state(self.curr_states[next_chain_name], new_state)
            seq[next_chain_name].append(t2s)
            self.curr_states[next_chain_name] = new_state
            self.curr_h_states[next_chain_name] = new_h_state

            self.timestamp += 1

        seq = dict(**seq)

        if len(self.labels) == 1:
            seq["class_label"] = list(self.labels.values())[0]
        else:
            for key, value in self.labels.items():
                seq["_".join(["class_label", value])] = key

        seq['event_time'] = [i for i in range(len(seq[next_chain_name]))]

        return seq


class BaseSchedule:
    def __init__(self, config):
        self.config = config
        self.chain_names = list(self.config.chains.keys())
        self.n_chains = len(self.chain_names)

    def get_next_chain(self, timestamp):
        raise NotImplemented


class SimpleSchedule(BaseSchedule):
    def get_next_chain(self, timestamp):
        return self.chain_names[int(timestamp % self.n_chains)]


class Config:
    def __init__(self, chain_confs, state_from, state_to, labeling_conf):
        """
        chain_confs = {
        "A": (n_states, sphere_R, noise),
        "B": (n_states, sphere_R, noise),
        "C": (n_states, sphere_R, noise),
        "D": (n_states, sphere_R, noise),
        }

        state_from = ["A", "B", "C"]

        state_to = ["B", "D", "D"]

        labeling_conf = {
        1 : {"A": 0.6},
        2 : {"B": 1.,
             "D": 0.2}
        }
        """
        assert len(state_from) == len(state_to)

        self.conj_from, self.conj_to = defaultdict(list), defaultdict(list)
        self.sampling_conf = dict()
        for f, t in zip(state_from, state_to):
            self.conj_from[t].append(f)
            self.conj_to[f].append(t)

        self.chain_confs = chain_confs
        self.chains = dict()
        for ch_name, ch_conf in self.chain_confs.items():
            n_states, sphere_r, noise = ch_conf
            if len(self.conj_from[ch_name]) == 0:
                n_h_states = 1
            else:
                n_h_states = np.prod([self.chain_confs[conj_ch][0] for conj_ch in self.conj_from[ch_name]])
            if len(self.conj_to[ch_name]) > 0:
                add_noise_tensor = True
            else:
                add_noise_tensor = False
            self.sampling_conf[ch_name] = {'n_states': n_states,
                                           'n_h_states': n_h_states,
                                           'dim': n_states ** 2,
                                           'sphere_r': sphere_r}
            self.chains[ch_name] = MarkovChain(n_states, sphere_r, n_h_states, add_noise_tensor, noise)

        self.labeling_conf = labeling_conf
        self.chain2label = defaultdict(list)
        self.class_assigners = dict()
        for key, chains in labeling_conf.items():
            for ch in chains:
                self.chain2label[ch].append(key)
            self.class_assigners[key] = PlaneClassAssigner(chains, self.sampling_conf)

    def get_chains(self):
        return deepcopy(self.chains)

    def save_assigners(self, file):
        for assigner_name, assigner in self.class_assigners.items():
            assigner.save_plane("_".join([file, str(assigner_name)]))

    def load_assigners(self, file):
        for assigner_name, assigner in self.class_assigners.items():
            assigner.load_plane("_".join([file, str(assigner_name)]))


class PlaneClassAssigner:
    """
    chains: {"B": 1.,
             "D": 0.2}
    sampling_conf: {"A": {'n_states': n_states,
                          'n_h_states': n_h_states,
                          'dim': n_states ** 2,
                          'sphere_r': sphere_r},
                    .
                    .
                    .
                   }
    """
    def __init__(self, chains_saturation, sampling_conf):
        self.chains_saturation = chains_saturation
        self.sampling_conf = sampling_conf
        self.v = None
        self.set_planes()

    def set_planes(self):
        self.v = dict()
        for ch in self.sampling_conf:
            if ch in self.chains_saturation:
                self.v[ch] = list()
                for i in range(self.sampling_conf[ch]['n_h_states']):
                    saturation_value = correct_round(self.sampling_conf[ch]['dim'] * self.chains_saturation[ch])

                    raw_vector = np.random.randn(self.sampling_conf[ch]['dim'])
                    saturation_vector = np.random.permutation(
                        [1. for _ in range(saturation_value)] +
                        [0. for _ in range(self.sampling_conf[ch]['dim'] - saturation_value)]
                    )
                    self.v[ch].append(raw_vector * saturation_vector)
            else:
                self.v[ch] = [np.zeros(self.sampling_conf[ch]['dim'])]

    def get_class(self, client_vector):
        max_prods, min_prods = list(), list()
        for ch, x in client_vector.items():
            if ch in self.chains_saturation:
                prod_max, prod_min = -float('inf'), float('inf')
                for xi, vi in zip(x, self.v[ch]):
                    prod = (xi * vi).sum()
                    prod_max = prod if prod > prod_max else prod_max
                    prod_min = prod if prod < prod_min else prod_min
                max_prods.append(prod_max)
                min_prods.append(prod_min)

        if sum(min_prods) > 0:
            c = 1
        elif sum(max_prods) < 0:
            c = 0
        else:
            c = None
        return c

    def save_plane(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.v, f)

    def load_plane(self, file):
        with open(file, "rb") as f:
            self.v = pickle.load(f)


class SphereSampler:
    def __init__(self, sampling_conf, class_assigners):
        self.sampling_conf = sampling_conf
        self.class_assigners = class_assigners

    def sample_tensor(self):
        tensor = dict()
        for ch in self.sampling_conf:
            tensor[ch] = list()
            for i in range(self.sampling_conf[ch]['n_h_states']):
                tensor[ch].append(np.random.randn(self.sampling_conf[ch]['dim']))
        return tensor

    def sample(self, labels):
        while True:
            flag = True
            candidate = self.norm(self.sample_tensor())
            for k, v in labels.items():
                if self.class_assigners[k].get_class(candidate) != v:
                    flag = False
                    break
            if flag:
                break

        candidate = self.reshape(candidate)
        return candidate

    def norm(self, client_tensor):
        for ch, t in client_tensor.items():
            r = self.sampling_conf[ch]['sphere_r']
            norm_t = list()
            for x in t:
                norm_x = x / np.sqrt((x**2).sum()) * r
                norm_t.append(norm_x)
            client_tensor[ch] = norm_t
        return client_tensor

    def reshape(self, client_tensor):
        for ch, t in client_tensor.items():
            n_states = self.sampling_conf[ch]['n_states']
            shaped_t = [self.softmax_norm(x.reshape(n_states, n_states)) for x in t]
            client_tensor[ch] = np.stack(shaped_t, axis=-1)
        return client_tensor

    @staticmethod
    def softmax_norm(matrix):
        return np.exp(matrix) / np.exp(matrix).sum(axis=-1, keepdims=True)


class MarkovChain:
    def __init__(self, n_states, sphere_r, n_h_states=1, add_noise_tensor=True, noise=0.):
        self.n_states = n_states
        self.n_h_states = n_h_states
        self.sphere_r = sphere_r
        self.transition_tensor = None

        self.add_noise_tensor = add_noise_tensor
        self.noise_tensor = None
        self.noise = noise

        self.state = None
        self.noise_state = None

        self.transition2state_matrix = np.arange(self.n_states**2).reshape(self.n_states, self.n_states)
        self.ready = False if self.transition_tensor is None else True

    def set_transition_tensor(self, transition_tensor=None):
        self.transition_tensor = transition_tensor
        condition = self.transition_tensor is not None and (self.noise_tensor is not None or not self.add_noise_tensor)
        self.ready = True if condition else False

    def set_noise_tensor(self, noise_tensor=None):
        self.noise_tensor = noise_tensor
        condition = self.transition_tensor is not None and (self.noise_tensor is not None or not self.add_noise_tensor)
        self.ready = True if condition else False

    def reset(self):
        assert self.ready
        self.state = np.random.choice(self.n_states)
        self.noise_state = np.random.choice(self.n_states)
        return self.state

    def next_state(self, h_state=None):
        assert self.ready
        if self.n_h_states <= 1:
            h_state = 0
        self.state = np.random.choice(self.n_states, p=self.transition_tensor[self.state, :, h_state])
        if self.noise_tensor is not None:
            self.noise_state = np.random.choice(self.n_states, p=self.noise_tensor[self.noise_state, :, h_state])
            state = self.state if np.random.rand() >= self.noise else self.noise_state
        else:
            state = self.state
        return state, self.state

    def transition2state(self, a, b):
        return self.transition2state_matrix[a, b]
