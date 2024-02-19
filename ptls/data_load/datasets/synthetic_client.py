import numpy as np
from collections import defaultdict
from copy import deepcopy


class SyntheticClient:
    def __init__(self, labels, config, schedule):
        self.labels = labels
        self.config = config
        self.schedule = schedule
        self.sampler = SphereSampler()
        self.chains = self.config.get_chains()
        self.gen_transition_tensors()

        self.timestamp = 0

    def gen_transition_tensors(self):
        for ch_name in self.chains:
            ch_transition_tensor = list()
            n_states = self.chains[ch_name].n_states
            n_h_states = self.chains[ch_name].n_h_states
            sphere_r = self.chains[ch_name].sphere_r
            assigner_names = self.config.chain2label[ch_name]
            labeling_tensor = [self.config.class_assigners[a_name].v[ch_name] for a_name in assigner_names]
            for h_state in range(n_h_states):
                label_vectors = [lv[h_state] for lv in labeling_tensor]
                matrix = self.sampler.sample(n_states, sphere_r, label_vectors).reshape(n_states, n_states)
                matrix = np.exp(matrix)/np.exp(matrix).sum(axis=-1, keepdims=True)
                ch_transition_tensor.append(matrix)
            self.chains[ch_name].set_transition_tensor(np.stack(ch_transition_tensor, axis=-1))

    def init_chains(self):
        pass

    def gen_seq(self, l=1000):
        seq = dict()
        self.init_chains()

        for i in range(l):
            next_chain = self.schedule.next_chain(self.timestamp)
            self.timestamp += 1


        return seq, self.labels


class Schedule:
    def init(self):
        pass


class Config:
    def __init__(self, chain_confs, state_from, state_to, labeling_conf):
        """
        chain_confs = {
        "A": (n_states, sphere_R),
        "B": (n_states, sphere_R),
        "C": (n_states, sphere_R),
        "D": (n_states, sphere_R),
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
            n_states, sphere_r = ch_conf
            if len(self.conj_from[ch_name]) == 0:
                n_h_states = 1
            else:
                n_h_states = np.prod([self.chain_confs[conj_ch][0] for conj_ch in self.conj_from[ch_name]])
            self.chains[ch_name] = MarkovChain(n_states, sphere_r, n_h_states)

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


class MarkovChain:
    def __init__(self, n_states, sphere_r, n_h_states=1, transition_tensor=None):
        self.n_states = n_states
        self.n_h_states = n_h_states
        self.sphere_r = sphere_r
        self.transition_tensor = transition_tensor
        self.state = None
        self.ready = False if self.transition_tensor is None else True

    def set_transition_tensor(self, transition_tensor=None):
        self.transition_tensor = transition_tensor
        self.ready = False if self.transition_tensor is None else True

    def reset(self):
        assert self.ready
        self.state = np.random.choice(self.n_states)

    def next_state(self, h_state=None):
        assert self.ready
        if self.n_h_states > 1:
            self.state = np.random.choice(self.n_states, p=self.transition_tensor[self.state, :, h_state])
        else:
            self.state = np.random.choice(self.n_states, p=self.transition_tensor[self.state, :, 0])
        return self.state


class SphereSampler:
    @staticmethod
    def sample(n_states, sphere_r, label_vectors=None):
        dim = n_states ** 2
        if label_vectors is not None:
            while True:
                x = np.random.randn(dim)
                flag = True
                for label_vector in label_vectors:
                    if x * label_vector < 0:
                        flag = False
                        break
                if flag:
                    break
        else:
            x = np.random.randn(dim)

        x = x / np.sqrt((x**2).sum())
        return x


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
