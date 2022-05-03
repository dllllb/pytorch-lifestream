class InfiniteBatchSampler:
    def __init__(self, epoch_size, batch_size):
        self.epoch_size = epoch_size
        self.batch_size = batch_size

        self._current_epoch = 0

    def __iter__(self):
        start_pos = self._current_epoch * self.epoch_size
        current_indexes = range(start_pos, start_pos + self.epoch_size)

        # TODO: Можно добавить shuffle, если синхронизировать его с InfiniteDataset

        batch = []
        for ix in current_indexes:
            batch.append(ix)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

        self._current_epoch += 1

    def __len__(self):
        return (self.epoch_size + self.batch_size - 1) // self.batch_size
