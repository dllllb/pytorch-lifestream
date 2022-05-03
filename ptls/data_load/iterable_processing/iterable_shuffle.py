import logging
from itertools import islice
import numpy as np
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


logger = logging.getLogger(__name__)


class IterableShuffle(IterableProcessingDataset):
    def __init__(self, buffer_size):
        """Remove records which are not in `relevant_ids`

        Args:
            buffer_size: buffer size in records
        """
        super().__init__()

        assert buffer_size > 1  # we will split buffer into two parts
        self._buffer_size = buffer_size

    def __iter__(self):
        source = iter(self._src)
        buffer = np.array([])
        while True:
            new_buffer_size = self._buffer_size - len(buffer)
            new_buffer_list = list(islice(source, new_buffer_size))
            new_buffer = np.empty(len(new_buffer_list), dtype=object)
            new_buffer[:] = new_buffer_list
            # print(f'Buffer shape: {buffer.shape}, {new_buffer.shape}')
            buffer = np.concatenate([buffer, new_buffer])
            if len(buffer) == 0:
                break

            window_size = min(len(buffer), self._buffer_size // 2)
            ix_for_choice = np.random.choice(len(buffer), window_size, replace=False)

            for ix in ix_for_choice:
                yield buffer[ix]

            mask_selected = np.zeros(len(buffer), dtype=np.bool)
            mask_selected[ix_for_choice] = True
            buffer = buffer[~mask_selected]
