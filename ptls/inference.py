import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict

from ptls.data_load import IterableChain, padded_collate
from ptls.data_load.filter_dataset import FilterDataset
from ptls.data_load.iterable_processing.category_size_clip import CategorySizeClip
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load.iterable_processing.target_extractor import FakeTarget
from ptls.train import score_model


def create_inference_dataloader(data: List[Dict],
                                num_workers: int = 0,
                                batch_size: int = 512,
                                category_names: List[str] = None,
                                category_max_size: Dict[str, int] = None):
    dataset = FilterDataset(
        data,
        post_processing=IterableChain(
            #FakeTarget(),
            FeatureFilter(keep_feature_names=category_names),
            CategorySizeClip(category_max_size, 1))
    )
    dataset = list(tqdm(iter(dataset)))

    return DataLoader(
        dataset=dataset,
        collate_fn=padded_collate,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
    )


def get_embeddings(
        data: List[Dict],
        model: torch.nn.Module,
        num_workers: int = 0,
        batch_size: int = 512,
        category_names: List[str] = None,
        category_max_size: Dict[str, int] = None):
    """
    Parameters
    ----------
     data: List[Dict]
        data for inference
     model: torch.nn.Module
        pytorch model for inference
     num_workers: int. Default: 0.
        The number of workers for dataloader. 0 = single-process loader
     batch_size: int. Default: 512.
        The number of samples in each batch.
    category_names: List[str]. Default: None.
        The names of features (keys in data)
    category_max_size: Dict[str, int]. Default: None.
        The max value of categorical features (embedding matrix size)
    """
    embeds, _ = score_model(
        model=model,
        valid_loader=create_inference_dataloader(data, num_workers, batch_size, category_names, category_max_size)
    )
    return embeds
