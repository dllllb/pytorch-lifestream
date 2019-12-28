# coding: utf-8
"""

"""
import numpy as np


class AbsSplit:
    def split(self, dates):
        raise NotImplementedError()


class NoSplit(AbsSplit):
    def split(self, dates):
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        return [date_range]


class SplitByCount(AbsSplit):
    def __init__(self, split_count, cnt_min, cnt_max):
        self.split_count = split_count
        self.cnt_min = cnt_min
        self.cnt_max = cnt_max

    def split(self, dates):
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        lengths = np.random.randint(self.cnt_min, self.cnt_max, self.split_count)
        available_start_pos = date_len - lengths
        start_pos = (np.random.rand(self.split_count) * available_start_pos).astype(int)
        return [date_range[s:s + l] for s, l in zip(start_pos, lengths)]


class SplitByWeeks(AbsSplit):
    def __init__(self, split_count, seed = 29, **kwargs):
        self.rs = np.random.RandomState(seed)
        self.split_count = split_count

    def split(self, dates):
        seq_len = len(dates)
        n_weeks = ((dates - dates[0])/np.timedelta64(1,'D') // 7).astype(np.int16)
        n_weeks_unique = np.unique(n_weeks)
        n_weeks_nunique = len(n_weeks_unique)

        if n_weeks_nunique < self.split_count: # split by days
            n_weeks = ((dates - dates[0])/np.timedelta64(1,'D') // 1).astype(np.int16)
            n_weeks_unique = np.unique(n_weeks)
            n_weeks_nunique = len(n_weeks_unique)

        if n_weeks_nunique < self.split_count: # split random
            n_weeks = np.arange(seq_len)
            n_weeks_unique = n_weeks
            n_weeks_nunique = len(n_weeks_unique)

        # devide n_weeks_unique for n_samples_from_class parts
        rand_perm = self.rs.permutation(n_weeks_nunique)
        n_weeks_idxes = [rand_perm[(i*n_weeks_nunique//self.split_count): \
                                   ((i+1)*n_weeks_nunique//self.split_count)] for i in range(self.split_count)]

        # select indexes correspond to each part of weeks
        # x_ij == 1 <=>  j-th elenemt of sequence correspond to n_weeks_unique[j]
        x = n_weeks.reshape(1,-1).repeat(n_weeks_nunique, axis = 0) == n_weeks_unique.reshape(-1,1).repeat(seq_len, axis = 1)
        n_byweeks_idxes = [x[one_week_idxes].sum(axis = 0).nonzero()[0] for one_week_idxes in n_weeks_idxes]

        return n_byweeks_idxes


def create(split_strategy, **params):
    cls = globals().get(split_strategy, None)
    if cls is None:
        raise AttributeError(f'Unknown split_strategy: "{split_strategy}"')
    if not issubclass(cls, AbsSplit):
        raise AttributeError(f'Wrong split_strategy: "{split_strategy}". This is not a splitter')
    if cls is AbsSplit:
        raise AttributeError(f'Wrong split_strategy: "{split_strategy}". Splitter can not be abstract')

    return cls(**params)
