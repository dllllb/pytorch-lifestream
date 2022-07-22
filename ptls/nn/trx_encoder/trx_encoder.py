import torch
from torch import nn as nn

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder.batch_norm import RBatchNorm, RBatchNormWithLens
from ptls.nn.trx_encoder.noisy_embedding import NoisyEmbedding
from ptls.nn.trx_encoder.float_positional_encoding import FloatPositionalEncoding
from ptls.nn.trx_encoder.scalers import scaler_by_name


class TrxEncoder(nn.Module):
    """Network layer which makes representation for single transactions

     Input is `PaddedBatch` with ptls-format dictionary, with feature arrays of shape (B, T)
     Output is `PaddedBatch` with transaction embeddings of shape (B, T, H)
     where:
        B - batch size, sequence count in batch
        T - sequence length
        H - hidden size, representation dimension

    Parameters
        embeddings:
            dict with categorical feature names.
            Values must be like this `{'in': dictionary_size, 'out': embedding_size}`
            These features will be encoded with lookup embedding table of shape (dictionary_size, embedding_size)
        numeric_values:
            dict with numerical feature names.
            Values must be a string with scaler_name.
            Possible values are: 'identity', 'sigmoid', 'log', 'year'.
            These features will be scaled with selected scaler and batch norm applyed.
        embeddings_noise (float):
            Noise level for embedding. `0` meens without noise

        norm_embeddings: keep default value for this parameter
        use_batch_norm_with_lens: keep default value for this parameter
        clip_replace_value: keep default value for this parameter
        positions: keep default value for this parameter

    Examples:
        >>> B, T = 5, 20
        >>> trx_encoder = TrxEncoder(
        >>>     embeddings={'mcc_code': {'in': 100, 'out': 5}},
        >>>     numeric_values={'amount': 'log'},
        >>> )
        >>> x = PaddedBatch(
        >>>     payload={
        >>>         'mcc_code': torch.randint(0, 99, (B, T)),
        >>>         'amount': torch.randn(B, T),
        >>>     },
        >>>     length=torch.randint(10, 20, (B,)),
        >>> )
        >>> z = trx_encoder(x)
        >>> assert z.payload.shape == (5, 20, 6)  # B, T, H
    """
    def __init__(self,
                 embeddings=None,
                 numeric_values=None,
                 embeddings_noise: float = 0,
                 norm_embeddings=None,
                 use_batch_norm_with_lens=False,
                 clip_replace_value=None,
                 positions=None,
                 ):
        super().__init__()
        self.scalers = nn.ModuleDict()

        self.use_batch_norm_with_lens = use_batch_norm_with_lens
        self.clip_replace_value = clip_replace_value if clip_replace_value else 'max'
        self.positions = positions if positions else {}

        for name, scaler_name in numeric_values.items():
            self.scalers[name] = torch.nn.Sequential(
                RBatchNormWithLens() if self.use_batch_norm_with_lens else RBatchNorm(),
                scaler_by_name(scaler_name),
            )

        self.embeddings = nn.ModuleDict()
        for emb_name, emb_props in embeddings.items():
            if emb_props.get('disabled', False):
                continue
            if emb_props['out'] == 0:
                continue
            self.embeddings[emb_name] = NoisyEmbedding(
                num_embeddings=emb_props['in'],
                embedding_dim=emb_props['out'],
                padding_idx=0,
                max_norm=1 if norm_embeddings else None,
                noise_scale=embeddings_noise,
            )

        self.pos = nn.ModuleDict()
        for pos_name, pos_params in self.positions.items():
            self.pos[pos_name] = FloatPositionalEncoding(**pos_params)

    def forward(self, x: PaddedBatch):
        processed = []

        for field_name, embed_layer in self.embeddings.items():
            feature = self._smart_clip(x.payload[field_name].long(), embed_layer.num_embeddings)
            processed.append(embed_layer(feature))

        for value_name, scaler in self.scalers.items():
            if self.use_batch_norm_with_lens:
                res = scaler(PaddedBatch(x.payload[value_name].float(), x.seq_lens))
            else:
                res = scaler(x.payload[value_name].float())
            processed.append(res)

        for pos_name, pe in self.pos.items():
            processed.append(pe(x.payload[pos_name].float()))

        out = torch.cat(processed, -1)
        return PaddedBatch(out, x.seq_lens)

    def _smart_clip(self, values, max_size):
        if self.clip_replace_value == 'max':
            return values.clip(0, max_size - 1)
        else:
            res = values.clone()
            res[values >= max_size] = self.clip_replace_value
            return res

    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        sz = len(self.scalers)
        sz += sum(econf.embedding_dim for econf in self.embeddings.values())
        sz += sum(pos_params.out_size for pos_params in self.pos.values())
        return sz

    @property
    def category_names(self):
        """Returns set of used feature names
        """
        return set([field_name for field_name in self.embeddings.keys()] +
                   [value_name for value_name in self.scalers.keys()] +
                   [pos_name for pos_name in self.pos.keys()]
                   )

    @property
    def category_max_size(self):
        """Returns dict with categorical feature names. Value is dictionary size
        """
        return {k: v.num_embeddings for k, v in self.embeddings.items()}
