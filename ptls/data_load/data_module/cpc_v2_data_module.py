"""
This class inherited from ColesDataModuleTrain. Just rewrited collate function
"""

"""
Creates data loaders for contrastive learning
Each record pass throw the sampler and splits into views of the main record
X - is feature_arrays, sub-samples of main event sequences according 
    split strategy: /dltranz/metric_learn/dataset/split_strategy.py

Input:            -> Output
dict of arrays    ->  X,               
client_1          -> [client_1_smpl_1, client_1_smpl_2, ..., client_1_smpl_n]
client_2          -> [client_2_smpl_1, client_2_smpl_2, ..., client_2_smpl_n]
...
client_k          -> [client_k_smpl_1, client_k_smpl_2, ..., client_k_smpl_n]
"""


import logging

import pytorch_lightning as pl
import torch

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ptls.trx_encoder import PaddedBatch
from ptls.data_load.data_module.coles_data_module import ColesDataModuleTrain

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    batch = [
        [PaddedBatch, PaddedBatch, ..., PaddedBatch],
        [PaddedBatch, PaddedBatch, ..., PaddedBatch],
        ...
        [PaddedBatch, PaddedBatch, ..., PaddedBatch]
    ]
    """
    split_count = len(batch[0])
    
    sequences = [defaultdict(list) for _ in range(split_count)]
    lengths = torch.zeros((len(batch), split_count), dtype=torch.int)
    
    for i, client_data in enumerate(batch):
        for j, subseq in enumerate(client_data):
            for k, v in subseq.items():
                sequences[j][k].append(v)
                lengths[i][j] = v.shape[0]
    
    # adding padds in each split
    padded_batch = []
    for j, seq in enumerate(sequences):
        for k, v in seq.items():
            seq[k] = torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
        padded_batch.append(PaddedBatch(seq, lengths[:, j].long()))
    
    return tuple(padded_batch)


class CpcV2DataModuleTrain(ColesDataModuleTrain):
    def __init__(self, conf, pl_module):
        super().__init__(conf, pl_module)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=collate_fn,
            shuffle=False if self._type == 'iterable' else True,
            num_workers=self.train_conf['num_workers'],
            batch_size=self.train_conf['batch_size'],
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=collate_fn,
            num_workers=self.valid_conf['num_workers'],
            batch_size=self.valid_conf['batch_size'],
        )
