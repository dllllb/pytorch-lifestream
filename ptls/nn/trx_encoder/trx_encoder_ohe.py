from typing import Dict

import torch as torch

from ptls.nn.trx_encoder.trx_encoder_base import TrxEncoderBase


class TrxEncoderOhe(TrxEncoderBase):
    def __init__(self,
                 embeddings: Dict[str, Dict] = None,
                 is_learnable: bool = False,
                 numeric_values: Dict[str, Dict] = None,
                 out_of_index: str = 'clip',
                 ):
        super().__init__(
            embeddings=embeddings,
            numeric_values=numeric_values,
            out_of_index=out_of_index,
        )

        self._embeddings = torch.nn.ModuleDict()
        for k, v in self.embeddings.items():
            self._embeddings[k] = torch.nn.Embedding(
                num_embeddings=v['in'],
                embedding_dim=v['in'],  # square
                padding_idx=0,
            )
            torch.nn.init.eye_(self._embeddings[k].weight.data)
            for p in self._embeddings[k].parameters():
                p.requires_grad = is_learnable
