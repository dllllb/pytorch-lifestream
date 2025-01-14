from typing import List, Dict

from torch.utils.data import DataLoader

from ptls.data_load import IterableChain, padded_collate_wo_target, collate_wo_target
from ptls.data_load.filter_dataset import FilterDataset
from ptls.data_load.iterable_processing import ToTorch, FilterNonArray, ISeqLenLimit


def inference_data_loader(
    data: List[Dict],
    max_seq_len: int = 10000,
    num_workers: int = 0,
    batch_size: int = 512,
    onnx: bool = False
) -> DataLoader:
    """
    Generate an inference data loader. The data loader will return a batch of sequences.

    Args:
        data: the dataset
        max_seq_len: the maximum sequence length. Only `max_seq_len` transaction will be taken from tail of sequence.
            Unlimited sequence length lead to out of memory error
        num_workers: the number of workers for the dataloader. Default: 0 - single-process loader.
        batch_size: the batch size. Default: 512. The number of samples (before splitting to subsequences) in
            each batch.
        onnx: flag for ONNX export. Default: False

    Returns:
        DataLoader
    """

    dataset = FilterDataset(
        data,
        post_processing=IterableChain(
            ToTorch(),
            FilterNonArray(),
            ISeqLenLimit(max_seq_len=max_seq_len),
        )
    )

    collate_fn = collate_wo_target if onnx else padded_collate_wo_target

    return DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
    )
