import numpy as np
from ptls.frames.coles.split_strategy import AbsSplit


class SampleSlices(AbsSplit):
    def __init__(self, long_split_count, pos_split_count, neg_split_count,
                 long_cnt_min, long_cnt_max,
                 short_cnt_min, short_cnt_max):
        self.long_split_count = long_split_count
        self.pos_split_count = pos_split_count
        self.neg_split_count = neg_split_count
        self.long_cnt_min = long_cnt_min
        self.long_cnt_max = long_cnt_max
        self.short_cnt_min = short_cnt_min
        self.short_cnt_max = short_cnt_max

    def split(self, dates):
        date_len = dates.shape[0]
        dates = np.arange(date_len)

        long_cnt_max = min(self.long_cnt_max, int(date_len/3))
        long_cnt_min = min(self.long_cnt_min, long_cnt_max-1)
        short_cnt_max = min(self.short_cnt_max, int(date_len/3))
        short_cnt_min = min(self.short_cnt_min, short_cnt_max-1)

        long_splits = {i: v for i, v in
                       enumerate(self.sub_split(dates, long_cnt_min, long_cnt_max, self.long_split_count))}

        positive_splits = {i: self.sub_split(split, short_cnt_min, short_cnt_max, self.pos_split_count)
                           for i, split in long_splits.items()}

        negative_splits = {i: self.neg_sub_split(dates, split, short_cnt_min, short_cnt_max, self.neg_split_count)
                           for i, split in long_splits.items()}

        return long_splits, positive_splits, negative_splits

    @staticmethod
    def sub_split(dates, a, b, n):
        date_len = dates.shape[0]

        lengths = np.random.randint(a, b + 1, n)
        available_start_pos = (date_len - lengths).clip(0, None)
        start_pos = (np.random.rand(n) * (available_start_pos + 1 - 1e-9)).astype(int)

        ix_sort = np.argsort(start_pos)
        return [dates[s:s + l] for s, l in zip(start_pos[ix_sort], lengths[ix_sort])]

    @staticmethod
    def neg_sub_split(dates, used_dates, a, b, n):
        date_len = dates.shape[0]
        left_start, right_end = 0, date_len - 1
        left_end, right_start = np.where(dates == used_dates[0])[0][0], np.where(dates == used_dates[-1])[0][0]
        left_len, right_len = left_end - left_start, right_end - right_start
        left_piece, right_piece = dates[left_start:left_end], dates[right_start:right_end]

        if left_len < a:
            return SampleSlices.sub_split(right_piece, a, b, n)
        elif right_len < a:
            return SampleSlices.sub_split(left_piece, a, b, n)
        else:
            p = left_len / (left_len + right_len)
            n_left = (np.random.rand(n) < p).sum()
            n_right = n - n_left

            neg_splits = list()
            if n_left:
                neg_splits.extend(SampleSlices.sub_split(left_piece, a, b, n_left))
            if n_right:
                neg_splits.extend(SampleSlices.sub_split(right_piece, a, b, n_right))
            return neg_splits
