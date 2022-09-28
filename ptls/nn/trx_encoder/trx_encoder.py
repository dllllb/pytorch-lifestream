import torch
import warnings

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder.batch_norm import RBatchNorm, RBatchNormWithLens
from ptls.nn.trx_encoder.noisy_embedding import NoisyEmbedding
from ptls.nn.trx_encoder.trx_encoder_base import TrxEncoderBase


class TrxEncoder(TrxEncoderBase):
    """Network layer which makes representation for single transactions

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
            Values must be like this `{'in': dictionary_size, 'out': embedding_size}`
            These features will be encoded with lookup embedding table of shape (dictionary_size, embedding_size)
            Values can be a `torch.nn.Embedding` implementation
        numeric_values:
            dict with numerical feature names.
            Values must be a string with scaler_name.
            Possible values are: 'identity', 'sigmoid', 'log', 'year'.
            These features will be scaled with selected scaler.
            Values can be `ptls.nn.trx_encoder.scalers.BaseScaler` implementatoin

            One field can have many scalers. In this case key become alias and col name should be in scaler.
            Check `TrxEncoderBase.numeric_values` for more details

        embeddings_noise (float):
            Noise level for embedding. `0` meens without noise
        emb_dropout (float):
            Probability of an element of embedding to be zeroed
        spatial_dropout (bool):
            Whether to dropout full dimension of embedding in the whole sequence

        use_batch_norm:
            True - All numerical values will be normalized after scaling
            False - No normalizing for numerical values
        use_batch_norm_with_lens:
            True - Respect seq_lens during batch_norm. Padding zeroes will be ignored
            False - Batch norm ower all time axis. Padding zeroes will included.

        orthogonal_init:
            if True then `torch.nn.init.orthogonal` applied
        linear_projection_size:
            Linear layer at the end will be added for non-zero value

        out_of_index:
            How to process a categorical indexes which are greater than dictionary size.
            'clip' - values will be collapsed to maximum index. This works well for frequency encoded categories.
                We join infrequent categories to one.
            'assert' - raise an error of invalid index appear.

        norm_embeddings: keep default value for this parameter
        clip_replace_value: Not useed. keep default value for this parameter
        positions: Not used. Keep default value for this parameter

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
                 use_batch_norm=True,
                 use_batch_norm_with_lens=False,
                 clip_replace_value=None,
                 positions=None,
                 emb_dropout=0,
                 spatial_dropout=False,
                 orthogonal_init=False,
                 linear_projection_size=0,
                 out_of_index: str = 'clip',
                 ):
        if clip_replace_value is not None:
            warnings.warn('`clip_replace_value` attribute is deprecated. Always "clip to max" used. '
                          'Use `out_of_index="assert"` to avoid categorical values clip', DeprecationWarning)

        if positions is not None:
            warnings.warn('`positions` is deprecated. positions is not used', UserWarning)

        if embeddings is None:
            embeddings = {}
        if numeric_values is None:
            numeric_values = {}

        noisy_embeddings = {}
        for emb_name, emb_props in embeddings.items():
            if emb_props.get('disabled', False):
                continue
            if emb_props['in'] == 0 or emb_props['out'] == 0:
                continue
            noisy_embeddings[emb_name] = NoisyEmbedding(
                num_embeddings=emb_props['in'],
                embedding_dim=emb_props['out'],
                padding_idx=0,
                max_norm=1 if norm_embeddings else None,
                noise_scale=embeddings_noise,
                dropout=emb_dropout,
                spatial_dropout=spatial_dropout,
            )

        super().__init__(
            embeddings=noisy_embeddings,
            numeric_values=numeric_values,
            out_of_index=out_of_index,
        )

        numerical_size = self.numerical_size
        if use_batch_norm and numerical_size > 0:
            if use_batch_norm_with_lens:
                self.numerical_batch_norm = RBatchNormWithLens(numerical_size)
            else:
                self.numerical_batch_norm = RBatchNorm(numerical_size)
        else:
            self.numerical_batch_norm = None

        if linear_projection_size > 0:
            self.linear_projection_head = torch.nn.Linear(super().output_size, linear_projection_size)
        else:
            self.linear_projection_head = None

        if orthogonal_init:
            for n, p in self.named_parameters():
                if n.startswith('embeddings.') and n.endswith('.weight'):
                    torch.nn.init.orthogonal_(p.data[1:])
                if n == 'linear_projection_head.weight':
                    torch.nn.init.orthogonal_(p.data)

    def forward(self, x: PaddedBatch):
        processed_embeddings = []
        processed_numerical = []

        for field_name in self.embeddings.keys():
            processed_embeddings.append(self.get_category_embeddings(x, field_name))

        for field_name in self.numeric_values.keys():
            processed_numerical.append(self.get_numeric_scaled(x, field_name))

        if len(processed_numerical):
            processed_numerical = torch.cat(processed_numerical, dim=2)
            if self.numerical_batch_norm is not None:
                processed_numerical = PaddedBatch(processed_numerical, x.seq_lens)
                processed_numerical = self.numerical_batch_norm(processed_numerical)
                processed_numerical = processed_numerical.payload
            processed_embeddings.append(processed_numerical)

        out = torch.cat(processed_embeddings, dim=2)

        if self.linear_projection_head is not None:
            out = self.linear_projection_head(out)
        return PaddedBatch(out, x.seq_lens)

    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        if self.linear_projection_head is not None:
            return self.linear_projection_head.out_features
        return super().output_size
