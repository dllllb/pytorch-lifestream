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


class SampleRandom(AbsSplit):
    def __init__(self, split_count, cnt_min, cnt_max):
        self.split_count = split_count
        self.cnt_min = cnt_min
        self.cnt_max = cnt_max

    def split(self, dates):
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        lengths = np.random.randint(self.cnt_min, self.cnt_max, self.split_count)
        splits = []
        for i, l in enumerate(lengths):
            rand_perm = np.random.permutation(date_len)
            splits.append(np.sort(date_range[rand_perm][:l]))
        return splits


class SplitRandom(AbsSplit):
    def __init__(self, split_count, cnt_min, cnt_max):
        self.split_count = split_count
        self.cnt_min = cnt_min
        self.cnt_max = cnt_max

    def split(self, dates):
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        split_indexes = np.random.randint(0, self.split_count, date_len)
        return [date_range[split_indexes == i][:self.cnt_max] for i in range(self.split_count)]


class SplitByWeeks(AbsSplit):
    def __init__(self, split_count, cnt_max, seed=29, week_length=7, **kwargs):
        self.rs = np.random.RandomState(seed)
        self.split_count = split_count
        self.cnt_max = cnt_max
        self.week_length = week_length

    def split(self, dates):
        seq_len = len(dates)
        n_weeks = ((dates - dates[0]).astype('timedelta64[D]') // self.week_length).astype(np.int16)
        n_weeks_unique = np.unique(n_weeks)
        n_weeks_nunique = len(n_weeks_unique)

        if n_weeks_nunique < self.split_count:  # split by days
            n_weeks = (dates - dates[0]).astype(np.int16)
            n_weeks_unique = np.unique(n_weeks)
            n_weeks_nunique = len(n_weeks_unique)

        if n_weeks_nunique < self.split_count:  # split random
            n_weeks = np.arange(seq_len)
            n_weeks_unique = n_weeks
            n_weeks_nunique = len(n_weeks_unique)

        # divide n_weeks_unique for split_count parts
        split_indexes = np.random.randint(0, self.split_count, n_weeks_nunique - self.split_count)
        week_range = np.arange(self.split_count, n_weeks_nunique)
        n_weeks_idxes = [week_range[split_indexes == i] for i in range(self.split_count)]
        n_weeks_idxes = [np.append(x, i) for i, x in enumerate(n_weeks_idxes)]

        # select indexes correspond to each part of weeks
        # x_ij == 1 <=>  j-th elenemt of sequence correspond to n_weeks_unique[i]
        x = n_weeks.reshape(1, -1).repeat(n_weeks_nunique, axis=0) == n_weeks_unique.reshape(-1, 1).repeat(seq_len,
                                                                                                           axis=1)
        n_byweeks_idxes = [x[one_week_idxes].sum(axis=0).nonzero()[0] for one_week_idxes in n_weeks_idxes]
        n_byweeks_idxes = [x[-self.cnt_max:] if len(x) > self.cnt_max else x for x in n_byweeks_idxes]
        return n_byweeks_idxes


class SampleSlices(AbsSplit):
    def __init__(self, split_count, cnt_min, cnt_max, short_seq_crop_rate=1.0, is_sorted=False):
        self.split_count = split_count
        self.cnt_min = cnt_min
        self.cnt_max = cnt_max
        self.short_seq_crop_rate = short_seq_crop_rate
        self.is_sorted = is_sorted

    def _validate_split_range(self, date_len):
        only_one_fold = date_len <= self.cnt_min and self.short_seq_crop_rate >= 1.0
        several_fold = int(date_len * self.short_seq_crop_rate) <= self.cnt_min and self.short_seq_crop_rate < 1.0
        return only_one_fold, several_fold

    def split(self, dates):
        date_len, date_range = dates.shape[0], np.arange(dates.shape[0])
        only_one_fold, several_fold = self._validate_split_range(date_len)
        if only_one_fold:
            return [date_range for _ in range(self.split_count)]

        cnt_min = self.cnt_min if not several_fold else int(date_len * self.short_seq_crop_rate)
        cnt_max = self.cnt_max if date_len > self.cnt_max else date_len

        lengths = np.random.randint(cnt_min, cnt_max + 1, self.split_count)
        available_start_pos = (date_len - lengths).clip(0, None)
        start_pos = (np.random.rand(self.split_count) * (available_start_pos + 1 - 1e-9)).astype(int)
        if not self.is_sorted:
            return [date_range[s:s + l] for s, l in zip(start_pos, lengths)]

        ix_sort = np.argsort(start_pos)
        return [date_range[s:s + l] for s, l in zip(start_pos[ix_sort], lengths[ix_sort])]


class SampleUniform(AbsSplit):
    """
    Sub samples with equal length = `seq_len`
    Start pos has fixed uniform distribution from sequence start to end with equal step
    |---------------------|       main sequence
    |------|              |        sub seq 1
    |    |------|         |        sub seq 2
    |         |------|    |        sub seq 3
    |              |------|        sub seq 4

    There is no random factor in this splitter, so sub sequences are the same every time
    Can be used during inference as test time augmentation
    """

    def __init__(self, split_count, seq_len, **_):
        self.split_count = split_count
        self.seq_len = seq_len

    def split(self, dates):
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        if date_len <= self.seq_len + self.split_count:
            return [date_range for _ in range(self.split_count)]

        start_pos = np.linspace(0, date_len - self.seq_len, self.split_count).round().astype(int)
        return [date_range[s:s + self.seq_len] for s in start_pos]


class SampleUniformBySplitCount(AbsSplit):
    """
    Split into n sections:
    |---------------------|       main sequence
    |------|              |        sub seq 1
    |       |------|      |        sub seq 2
    |              |------|        sub seq 3
    There is no random factor in this splitter, so sub sequences are the same every time
    Can be used during inference as test time augmentation
    """

    def __init__(self, split_count, **_):
        self.split_count = split_count

    def split(self, dates):
        date_range = np.arange(dates.shape[0])
        return np.array_split(date_range, self.split_count)


class SplitByNextNearestTime(AbsSplit):
    """
    Split into subsequences so that each next one is close in time to the previous one.
    """

    def __init__(self, split_count, cnt_min, cnt_max, margin=0):
        self.split_count = split_count
        self.cnt_min = cnt_min
        self.cnt_max = cnt_max
        self.margin = margin

    def split(self, dates):
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        cnt_max = min(self.cnt_max, date_len)
        cnt_min = self.cnt_min

        lengths = np.random.randint(cnt_min, cnt_max, self.split_count)

        prev_len = lengths[0]
        prev_pos = np.random.randint(0, date_len - prev_len)

        splits = [date_range[prev_pos: prev_pos + prev_len]]

        for i in range(1, self.split_count):
            curr_len = lengths[i]

            curr_start = max(0, prev_pos - curr_len - self.margin)
            curr_end = min(date_len - curr_len, prev_pos + prev_len + self.margin)

            curr_pos = np.random.randint(curr_start, curr_end)
            splits.append(date_range[curr_pos: curr_pos + curr_len])

            prev_pos, prev_len = curr_pos, curr_len

        return splits


class SplitByNearestTime(AbsSplit):
    """
    Split into subsequences close in time.
    """

    def __init__(self, split_count, cnt_min, cnt_max, margin=0):
        self.split_count = split_count
        self.cnt_min = cnt_min
        self.cnt_max = cnt_max
        self.margin = margin

    def split(self, dates):
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        cnt_max = min(date_len, self.cnt_max)
        cnt_min = self.cnt_min

        lengths = np.random.randint(cnt_min, cnt_max, self.split_count)

        len_start = lengths[0]
        pos_start = np.random.randint(0, date_len - len_start)
        splits = [date_range[pos_start: pos_start + len_start]]

        for i in range(self.split_count - 1):
            curr_len = lengths[i]
            curr_start = max(0, pos_start - curr_len - self.margin)
            curr_end = min(pos_start + len_start + self.margin, date_len - curr_len)

            curr_pos = np.random.randint(curr_start, curr_end)

            splits.append(date_range[curr_pos: curr_pos + curr_len])

        return splits


class CutByDays(AbsSplit):
    def __init__(self, first_date, last_date):
        self.days_arange = np.arange(first_date, last_date + 1)

    def split(self, dates):
        all_indexes = np.arange(len(dates))
        left_indexes = np.searchsorted(dates.astype(np.int32), self.days_arange)
        indexes = [all_indexes[:x] for x in left_indexes]
        return indexes


def create(split_strategy, **params):
    cls = globals().get(split_strategy, None)
    if cls is None:
        raise AttributeError(f'Unknown split_strategy: "{split_strategy}"')
    if not issubclass(cls, AbsSplit):
        raise AttributeError(f'Wrong split_strategy: "{split_strategy}". This is not a splitter')
    if cls is AbsSplit:
        raise AttributeError(f'Wrong split_strategy: "{split_strategy}". Splitter can not be abstract')

    return cls(**params)
