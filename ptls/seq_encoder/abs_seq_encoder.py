import torch


class AbsSeqEncoder(torch.nn.Module):
    def __init__(self, is_reduce_sequence=True):
        super().__init__()
        self._is_reduce_sequence = is_reduce_sequence

    @property
    def is_reduce_sequence(self):
        return self._is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self._is_reduce_sequence = value

    @property
    def embedding_size(self):
        raise NotImplementedError()
