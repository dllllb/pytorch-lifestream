import warnings
from typing import Dict

import torch as torch
from ptls.nn.trx_encoder.trx_encoder import TrxEncoder


class TrxEncoderOhe(TrxEncoder):
    """Network layer which makes representation for single transactions
    Based on `ptls.nn.trx_encoder.TrxEncoder` with embedding one-hot initialization.

     Input is `PaddedBatch` with ptls-format dictionary, with feature arrays of shape (B, T)
     Output is `PaddedBatch` with transaction embeddings of shape (B, T, H)
     where:
        B - batch size, sequence count in batch
        T - sequence length
        H - hidden size, representation dimension

    `ptls.nn.trx_encoder.noisy_embedding.NoisyEmbedding` implementation are used for categorical features.

    Parameters
        embeddings:
            dict with categorical feature names.
            Values must be like this `{'in': dictionary_size}`
            These features will be encoded with lookup embedding table of shape (dictionary_size, embedding_size)
        is_learnable:
            correspond `requires_grad` attribute of embeddings parameters.
            Linear projection always learnable (if exists)
        numeric_values:
            dict with numerical feature names.
            Values must be a string with scaler_name.
            Possible values are: 'identity', 'sigmoid', 'log', 'year'.
            These features will be scaled with selected scaler.
            Values can be `ptls.nn.trx_encoder.scalers.BaseScaler` implementatoin

            One field can have many scalers. In this case key become alias and col name should be in scaler.
            Check `TrxEncoderBase.numeric_values` for more details

        use_batch_norm:
            True - All numerical values will be normalized after scaling
            False - No normalizing for numerical values
        use_batch_norm_with_lens:
            True - Respect seq_lens during batch_norm. Padding zeroes will be ignored
            False - Batch norm ower all time axis. Padding zeroes will included.

        linear_projection_size:
            Linear layer at the end will be added for non-zero value

        out_of_index:
            How to process a categorical indexes which are greater than dictionary size.
            'clip' - values will be collapsed to maximum index. This works well for frequency encoded categories.
                We join infrequent categories to one.
            'assert' - raise an error of invalid index appear.

    Examples:
        >>> B, T = 5, 20
        >>> trx_encoder = TrxEncoderOhe(
        >>>     embeddings={'mcc_code': {'in': 100}},
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
        >>> assert z.payload.shape == (5, 20, 100)  # B, T, H
    """

    def __init__(self,
                 embeddings: Dict[str, Dict] = None,
                 is_learnable: bool = False,
                 numeric_values: Dict[str, Dict] = None,
                 use_batch_norm=True,
                 use_batch_norm_with_lens=False,
                 linear_projection_size=0,
                 out_of_index: str = 'clip',
                 ):

        for k, v in embeddings.items():
            new_out = v['in'] - 1
            if 'out' in v:
                if v['out'] != new_out:
                    warnings.warn(f'Embedding out correction. Embeddings out for "{k}" should be equal `in - 1`, '
                                  f'found {v["out"]}, should be {new_out}. '
                                  f'Set correct value for `embeddings.{k}.out` or drop it to use automatic output set',
                                  UserWarning)
            v['out'] = new_out

        super().__init__(
            embeddings=embeddings,
            numeric_values=numeric_values,
            out_of_index=out_of_index,
            use_batch_norm=use_batch_norm,
            use_batch_norm_with_lens=use_batch_norm_with_lens,
            linear_projection_size=linear_projection_size,
            embeddings_noise=0,
            emb_dropout=0,
            spatial_dropout=False,
            orthogonal_init=False,
        )

        for n, p in self.named_parameters():
            if n.startswith('embeddings.') and n.endswith('.weight'):
                torch.nn.init.eye_(p.data[1:])
                p.requires_grad = is_learnable
