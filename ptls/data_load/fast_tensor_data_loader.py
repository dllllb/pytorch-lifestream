from typing import Optional
import torch


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, 
                 batch_size: int = 32, 
                 shuffle: bool = False, 
                 post_process_func: Optional[callable] = None):
        """
        Initializes a FastTensorDataLoader.

        Args:
            *tensors: Tensors to store. Must have the same length at dimension 0.
            batch_size: Batch size to load. Defaults to 32.
            shuffle: If True, shuffle the data in-place whenever an iterator is created 
                out of this object. Defaults to False.
            post_process_func: If not None, apply function to the resulting batch.

        Returns:
            FastTensorDataLoader: A FastTensorDataLoader instance.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)

        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.post_process_func = post_process_func

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if len(self.tensors) > 1:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        else:
            batch = self.tensors[0][self.i:self.i+self.batch_size]

        self.i += self.batch_size

        if self.post_process_func is not None:
            batch = self.post_process_func(batch)

        return batch

    def __len__(self):
        return self.n_batches
