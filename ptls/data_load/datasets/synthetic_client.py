import numpy as np
from collections import defaultdict
from copy import deepcopy
import pickle
from .synthetic_utils import correct_round, sign, norm_vector


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
        self.backup_v = None
        self.backup_sat = None
        self.h_saturation_impact, self.n_sat, self.saturation_factor = 0, 0, 0
        self.set_planes()

        assert len(self.chains_saturation) <= 2
        assert min([self.sampling_conf[ch]['n_h_states'] for ch in self.chains_saturation]) == 1

    def set_planes(self):
        self.v = dict()
        self.backup_v = dict()
        self.backup_sat = dict()
        for ch in self.chains_saturation:
            self.v[ch] = list()
            self.backup_v[ch] = list()

            self.n_sat += self.chains_saturation[ch]
            if self.sampling_conf[ch]['n_h_states'] > 1:
                self.h_saturation_impact = self.chains_saturation[ch]

            self.backup_sat[ch] = np.random.permutation(np.arange(self.sampling_conf[ch]['dim']))
            saturation_value = correct_round(self.sampling_conf[ch]['dim'] * self.chains_saturation[ch])
            saturation_vector = np.where(self.backup_sat[ch] < saturation_value, 1, 0)

            for i in range(self.sampling_conf[ch]['n_h_states']):
                raw_vector = np.random.randn(self.sampling_conf[ch]['dim'])
                vector = norm_vector(raw_vector * saturation_vector)
                self.backup_v[ch].append(raw_vector)
                self.v[ch].append(vector)

        self.saturation_factor = self.h_saturation_impact / self.n_sat

    def check_class(self, client_vector, class_label):
        resample_dict = dict()
        mono_ch = list()
        multi_ch = None
        curr_prod = np.zeros(1)
        for ch, x in client_vector.items():
            if ch in self.chains_saturation:

                num_vectors = len(x)
                if num_vectors > 1:
                    multi_ch = ch
                    curr_prod = np.tile(curr_prod, num_vectors)
                else:
                    mono_ch.append(ch)

                norm_x = [norm_vector(a) for a in x]
                x_ = np.stack(norm_x, axis=0)
                v_ = np.stack(self.v[ch], axis=0)
                prod = (x_ * v_).sum(axis=1)
                curr_prod = curr_prod + prod
        class_sign = 1 if class_label == 1 else -1
        total = len(curr_prod)
        corr = sign(curr_prod)
        num_corr = (corr == class_sign).sum()
        if total == num_corr:
            return dict()
        elif num_corr / total < self.saturation_factor or self.saturation_factor == 0:
            for ch in mono_ch:
                resample_dict[ch] = [0]
        else:
            resample_dict[multi_ch] = np.where(corr != class_sign)[0]
        return resample_dict

    def save_plane(self, file):
        with open(file + "_v", "wb") as f:
            pickle.dump(self.backup_v, f)
        with open(file + "_sat", "wb") as f:
            pickle.dump(self.backup_sat, f)

    def load_plane(self, file):
        with open(file + "_v", "rb") as f:
            self.backup_v = pickle.load(f)
        with open(file + "_sat", "rb") as f:
            self.backup_sat = pickle.load(f)

        self.v = dict()
        for ch in self.chains_saturation:
            saturation_value = correct_round(self.sampling_conf[ch]['dim'] * self.chains_saturation[ch])
            saturation_vector = np.where(self.backup_sat[ch] < saturation_value, 1, 0)
            self.v[ch] = list()
            for vector in self.backup_v[ch]:
                self.v[ch].append(norm_vector(vector * saturation_vector))


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

    def resample(self, candidate, resample_dict):
        for ch in candidate:
            if ch in resample_dict:
                for i, vector in enumerate(candidate[ch]):
                    if i in resample_dict[ch]:
                        new_vector = np.random.randn(vector.shape[0])
                        candidate[ch][i] = new_vector
        candidate = self.norm(candidate)
        return candidate

    def sample(self, labels):
        n_try = 0
        candidate = self.norm(self.sample_tensor())
        while True:
            flag = True
            for k, v in labels.items():
                resample_dict = self.class_assigners[k].check_class(candidate, v)
                if len(resample_dict):
                    candidate = self.resample(candidate, resample_dict)
                    flag = False
                    n_try += 1
                    break
            if flag:
                break
            elif n_try == 100:
                n_try = 0
                candidate = self.norm(self.sample_tensor())

        candidate = self.reshape(candidate)
        return candidate

    def norm(self, client_tensor):
        for ch, t in client_tensor.items():
            r = self.sampling_conf[ch]['sphere_r']
            norm_t = list()
            for x in t:
                norm_x = norm_vector(x) * r
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
