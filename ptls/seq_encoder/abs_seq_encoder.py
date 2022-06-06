import torch


class AbsSeqEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.is_reduce_sequence = True

    @property
    def is_reduce_sequence(self):
        return self._is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self._is_reduce_sequence = value

    @property
    def category_max_size(self):
        raise NotImplementedError()

    @property
    def category_names(self):
        raise NotImplementedError()

    @property
    def embedding_size(self):
        raise NotImplementedError()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
