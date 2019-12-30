import logging
import pickle

from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

from tools.shape_generator import ShapeGenerator

logger = logging.getLogger(__name__)


def save_dataset(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f'Dataset with {len(data)} items saved to "{path}"')


def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f'Dataset with {len(data)} items loaded from "{path}"')
    return data


def create_supervised(size, generator_params):
    logger.info(f'Supervised: size = {size}')
    data = [ShapeGenerator(**generator_params) for _ in range(size)]
    buffer = [obj.get_buffer() for obj in tqdm(data)]
    return buffer


def create_metric_learning(size, sample_count, generator_params):
    logger.info(f'Metric learning: size = {size}, sample_count = {sample_count}')
    data = [ShapeGenerator(**generator_params) for _ in range(size)]
    buffer = [[(obj.get_buffer()[0], item) for _ in range(sample_count)]
              for item, obj in enumerate(tqdm(data))]
    return buffer


class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class ImageMetricLearningDataset(Dataset):
    def __init__(self, data, sample_count):
        self.data = data
        self.sample_count = sample_count

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        obj = self.data[item]
        return obj[:self.sample_count]


class ImageMetricLearningDatasetOnline(Dataset):
    def __init__(self, data, sample_count):
        self.data = data
        self.sample_count = sample_count

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        obj = self.data[item]
        data = [(obj.get_buffer()[0], item) for _ in range(self.sample_count)]
        return data
