import numpy as np


class ListSplitter:
    """
    Collect list of objects and split it on train and valid (if possible)
    Returns list of objects for train and valid
    """
    def __init__(self, object_list, valid_size=0.5, is_sort=True, seed=None):
        self.object_list = object_list
        self.valid_size = valid_size
        self.is_sort = is_sort
        self.seed = seed

        self._n = len(self.object_list)
        self._train_files = None
        self._valid_files = None

        self.split()
        self.sort()

    @property
    def train(self):
        return self._train_files

    @property
    def valid(self):
        return self._valid_files

    def size_select(self):
        if self.valid_size == 0.0:
            return self._n, 0

        _valid = self._n * self.valid_size
        if _valid < 1.0:
            _valid = 1
        else:
            _valid = int(round(_valid))

        _train = self._n - _valid
        if _train == 0:
            raise AttributeError(', '.join([
                'Incorrect train size',
                f'N: {self._n}',
                f'train: {_train}',
                f'valid: {_valid}',
                f'rate: {self.valid_size:.3f}',
            ]))
        return _train, _valid

    def split(self):
        train_size, valid_size = self.size_select()

        if valid_size == 0:
            self._train_files = self.object_list
            self._valid_files = None
        else:
            valid_ix = np.arange(self._n)
            rs = np.random.RandomState(self.seed)
            valid_ix = set(rs.choice(valid_ix, size=valid_size, replace=False).tolist())

            self._train_files = [rec for i, rec in enumerate(self.object_list) if i not in valid_ix]
            self._valid_files = [rec for i, rec in enumerate(self.object_list) if i in valid_ix]

    def sort(self):
        if not self.is_sort:
            return
        if self._train_files is not None:
            self._train_files = sorted(self._train_files)
        if self._valid_files is not None:
            self._valid_files = sorted(self._valid_files)
