import torch


class AllTimeShuffle:
    """Shuffle all transactions in event sequence
    """
    def __init__(self, event_time_name='event_time'):
        self.event_time_name = event_time_name

    @staticmethod
    def get_perm_ix(event_time):
        n = len(event_time)
        return torch.randperm(n)

    def __call__(self, x):
        ix = self.get_perm_ix(x[self.event_time_name])
        new_x = {k: v[ix] for k, v in x.items()}
        return new_x
