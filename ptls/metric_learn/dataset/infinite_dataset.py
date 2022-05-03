import logging
from glob import glob
from itertools import islice

from tqdm.autonotebook import tqdm

from ptls.data_load import read_spark_dataset_gen, read_dataset_mthread
from torch.utils.data import Dataset

from ptls.util import get_data_files

logger = logging.getLogger(__name__)


class InfiniteDataset(Dataset):
    # TODO: можно добавить чтение заранее, чтобы подгрузка новых данных не начиналась в середине эпохи
    def __init__(self, dataset_params, prepare_gen, max_file_read=None, progress=False):
        self.dataset_params = dataset_params
        self.prepare_gen = prepare_gen
        self.max_file_read = max_file_read
        self.n_workers = dataset_params.get('n_workers', 1)
        self.file_batch_size = dataset_params.get('file_batch_size', self.n_workers)
        if self.max_file_read is not None and self.n_workers > 1:
            self.n_workers = 1
            logger.warning('max_file_read mode is active. n_workers reduced to 1')
        self.progress = progress

        self._file_list = get_data_files(dataset_params)
        self._next_file = 0

        self._current_buffer = None
        self._global_start_pos = 0
        self._global_next_file_pos = 0

    def __len__(self):
        raise AssertionError('Can not estimate size of infinite dataset')

    def __getitem__(self, item):
        if item >= self._global_next_file_pos:
            self._read_next_files()

        if item >= self._global_next_file_pos:
            raise IndexError(f"Can not go to future: "
                             f"{item} not in [{self._global_start_pos} : {self._global_next_file_pos}]")
        if item < self._global_start_pos:
            raise IndexError(f"Can not go to past: "
                             f"{item} not in [{self._global_start_pos} : {self._global_next_file_pos}]")

        return self._current_buffer[item - self._global_start_pos]

    def _read_next_files(self):
        file_log_num = [self._next_file + i for i in range(self.file_batch_size)]
        file_real_num = [i % len(self._file_list) for i in file_log_num]
        files = [self._file_list[i] for i in file_real_num]

        gen = read_dataset_mthread(files, self.n_workers, self.prepare_gen)
        gen = tqdm(gen, desc=f'Read from files {file_log_num}/{len(self._file_list)} (real file nums {file_real_num})',
                   disable=not self.progress)
        if self.max_file_read is not None:
            gen = islice(gen, self.max_file_read)

        logger.info(f'Reading files {file_log_num}/{len(self._file_list)} (real file nums {file_real_num}) ...')
        data = [x for x in gen]
        self._next_file += len(files)
        logger.info(f'Loaded {len(data)} rows from {len(files)} files')

        self._current_buffer = data
        self._global_start_pos = self._global_next_file_pos
        self._global_next_file_pos = self._global_start_pos + len(data)

    def skip_first(self, size):
        while size >= self._global_next_file_pos:
            self._read_next_files()

        buffer_new_start = size - self._global_start_pos
        skipped_data = self._current_buffer[:buffer_new_start]
        self._current_buffer = self._current_buffer[buffer_new_start:]
        self._global_start_pos = 0  # = _global_start_pos - size + buffer_new_start
        self._global_next_file_pos -= size
        return skipped_data
