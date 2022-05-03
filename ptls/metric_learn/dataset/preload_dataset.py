import logging
from glob import glob
from itertools import islice

from tqdm.autonotebook import tqdm

from ptls.data_load import read_spark_dataset_gen, read_dataset_mthread
from torch.utils.data import Dataset, DataLoader

from ptls.util import get_data_files, cycle_block_iterator

logger = logging.getLogger(__name__)


class PreloadDataset(Dataset):
    # TODO: можно добавить чтение заранее, чтобы подгрузка новых данных не начиналась в середине эпохи
    def __init__(self, params, prepare_gen, max_file_read=None, progress=False):
        self.dataset_params = params['dataset']
        self.prepare_gen = prepare_gen
        self.max_file_read = max_file_read
        self.n_workers = self.dataset_params.get('n_workers', 1)
        self.file_batch_size = self.dataset_params.get('file_batch_size', self.n_workers)
        if self.max_file_read is not None and self.n_workers > 1:
            self.n_workers = 1
            logger.warning(f'max_file_read mode is active. n_workers reduced to 1')
        self.progress = progress

        self._file_list = get_data_files(self.dataset_params)
        self._prepare_file_iterator()
        self.data = []
        self.train_epoch_size = params['params.train.epoch_size']
        self.n_epoch = -1

    def __len__(self):
        return self.train_epoch_size

    def __getitem__(self, item):
        return self.data[item]

    def _prepare_file_iterator(self):
        self.file_iterator = cycle_block_iterator(
            iter(range(len(self._file_list))),
            self.file_batch_size
        )

    def _read_next_files(self):
        file_real_num = next(self.file_iterator)
        files = [self._file_list[i] for i in file_real_num]

        gen = read_dataset_mthread(files, self.n_workers, self.prepare_gen)
        gen = tqdm(gen, desc=f'Read from files {file_real_num}/{len(self._file_list)}',
                   disable=not self.progress)
        if self.max_file_read is not None:
            gen = islice(gen, self.max_file_read)

        logger.info(f'Reading files {file_real_num}/{len(self._file_list)} ...')
        data = [x for x in gen]

        logger.info(f'Loaded {len(data)} rows from {len(files)} files')
        self.data += data

        return files

    def prepare_epoch(self):
        self.n_epoch += 1
        if self.n_epoch > 0:
            self.data = self.data[self.train_epoch_size:]
        while len(self.data) < self.train_epoch_size:
            self._read_next_files()
        logger.info(f'Epoch {self.n_epoch}, {len(self.data)} rows in buffer')

    def skip_first(self, size):
        readed_files = []
        while size > len(self.data):
            readed_files += self._read_next_files()

        skipped_data = self.data[:size]
        self.data = self.data[size:]
        self._file_list = [x for x in self._file_list if x not in readed_files]
        self._prepare_file_iterator()

        return skipped_data


class PreloadDataLoader(DataLoader):
    def prepare_epoch(self):
        self.preload_dataset.prepare_epoch()
