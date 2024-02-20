import numpy as np
from collections import defaultdict
from copy import deepcopy
import pickle


class SyntheticClient:
    def __init__(self, labels, config, schedule):
        self.labels = labels
        self.config = config
        self.schedule = schedule
        self.sampler = SphereSampler()
        self.chains = self.config.get_chains()

        self.curr_states = None
        self.timestamp = 0

        self.norm_labels = dict()
        for y in self.labels:
            self.norm_labels[y] = 1 if self.labels[y] else -1

        self.set_transition_tensors()

    def gen_set_transition_tensor(self, ch_name):
        ch_transition_tensor = list()
        n_states = self.chains[ch_name].n_states
        n_h_states = self.chains[ch_name].n_h_states
        sphere_r = self.chains[ch_name].sphere_r
        assigner_names = self.config.chain2label[ch_name]
        labeling_tensors = [self.config.class_assigners[a_name].v[ch_name] for a_name in assigner_names]
        norm_labels = [self.norm_labels[y] for y in assigner_names]
        for h_state in range(n_h_states):
            label_vectors = [lv[h_state] * norm for lv, norm in zip(labeling_tensors, norm_labels)]
            matrix = self.sampler.sample(n_states, sphere_r, label_vectors).reshape(n_states, n_states)
            matrix = np.exp(matrix) / np.exp(matrix).sum(axis=-1, keepdims=True)
            ch_transition_tensor.append(matrix)
        return np.stack(ch_transition_tensor, axis=-1)

    def set_transition_tensors(self):
        for ch_name in self.chains:
            tensor = self.gen_set_transition_tensor(ch_name)
            self.chains[ch_name].set_transition_tensor(tensor)

            if self.chains[ch_name].add_noise_tensor:
                noise_tensor = self.gen_set_transition_tensor(ch_name)
                self.chains[ch_name].set_noise_tensor(noise_tensor)

    def init_chains(self):
        self.curr_states = dict()
        for ch in self.chains:
            self.curr_states[ch] = self.chains[ch].reset()

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
                    h_state += skip * self.curr_states[ch]
                    skip *= self.chains[ch].n_states
            else:
                h_state = None

            new_state = self.chains[next_chain_name].next_state(h_state)
            t2s = self.chains[next_chain_name].transition2state(self.curr_states[next_chain_name], new_state)
            seq[next_chain_name].append(t2s)
            self.curr_states[next_chain_name] = new_state

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
        1 : ("A",),
        2 : ("B", "D")
        }
        """
        assert len(state_from) == len(state_to)

        self.conj_from, self.conj_to = defaultdict(list), defaultdict(list)
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
            self.chains[ch_name] = MarkovChain(n_states, sphere_r, n_h_states, add_noise_tensor, noise)

        self.labeling_conf = labeling_conf
        self.chain2label = defaultdict(list)
        self.class_assigners = dict()
        for key, chains in labeling_conf.items():
            for ch in chains:
                self.chain2label[ch].append(key)
            n_states = (self.chains[ch].n_states for ch in chains)
            n_h_states = (self.chains[ch].n_h_states for ch in chains)
            self.class_assigners[key] = PlaneClassAssigner(chains, n_states, n_h_states)

    def get_chains(self):
        return deepcopy(self.chains)

    def save_assigners(self, file):
        for assigner_name, assigner in self.class_assigners.items():
            assigner.save_tensor("_".join([file, str(assigner_name)]))

    def load_assigners(self, file):
        for assigner_name, assigner in self.class_assigners.items():
            assigner.load_tensor("_".join([file, str(assigner_name)]))


class PlaneClassAssigner:
    """
    chains: ("A", "B")
    n_states: (4, 8)
    n_hidden_states: (1, 4)
    """
    def __init__(self, chains, n_states, n_hidden_states):
        self.chains = chains
        self.n_states = n_states
        self.n_hidden_states = n_hidden_states
        self.v = None
        self.set_planes()

    def set_planes(self):
        self.v = dict()
        for ch, n_states, n_hidden_states in zip(self.chains, self.n_states, self.n_hidden_states):
            ch_v = list()
            dim = n_states ** 2
            for i in range(n_hidden_states):
                v = np.random.randn(dim)
                v = self.norm_vector(v)
                ch_v.append(v)
            self.v[ch] = ch_v

    @staticmethod
    def norm_vector(v):
        return v / np.sqrt((v ** 2).sum())

    def get_class(self, x, h_state_n=0, ch=None):
        assert ch is not None or len(self.chains) == 1
        ch = self.chains[0] if ch is None else ch
        v = self.v[ch][h_state_n]
        c = 1 if (v * x).sum() > 0 else 0
        return c

    def save_tensor(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.v, f)

    def load_tensor(self, file):
        with open(file, "rb") as f:
            self.v = pickle.load(f)


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
        return state

    def transition2state(self, a, b):
        return self.transition2state_matrix[a, b]


class SphereSampler:
    @staticmethod
    def sample(n_states, sphere_r, label_vectors=None):
        dim = n_states ** 2
        if label_vectors is not None:
            while True:
                x = np.random.randn(dim)
                flag = True
                for label_vector in label_vectors:
                    if (x * label_vector).sum() < 0:
                        flag = False
                        break
                if flag:
                    break
        else:
            x = np.random.randn(dim)

        x = x / np.sqrt((x**2).sum()) * sphere_r
        return x
