import logging
import random
import math
import torch
from torch import nn

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model,
                 use_start_random_shift=True,
                 max_len=5000,
                 ):
        super(PositionalEncoding, self).__init__()
        self.use_start_random_shift = use_start_random_shift
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        if self.training and self.use_start_random_shift:
            start_pos = random.randint(0, self.max_len - T)
        else:
            start_pos = 0
        x = x + self.pe[:, start_pos:start_pos + T]
        return x


class TransformerEncoder(AbsSeqEncoder):
    """Used torch implementation of transformer
    Based on `torch.nn.TransformerEncoder`

    Parameters
        input_size:
            input embedding size.
            Equals intermediate and output layer size cause transformer don't change vector dimentions
        train_starter:
            'randn' or 'zeros'
            Which token used for CLS token, random learnable or zeros fixed
        shared_layers:
            True - then the same weights used for all `n_layers`.
            False - `n_layers` used different weights
        n_heads:
            The number of heads in the multiheadattention models
        dim_hidden:
            The dimension of the feedforward network model
        dropout:
            The dropout value
        n_layers:
            The number of sub-encoder-layers in the encoder
        use_positional_encoding (bool):
            Use or not positional encoding
        use_start_random_shift (bool):
            True - starting pos of positional encoding randomly shifted when training
            This allow to train transformer with all range of positional encoding values
            False - starting pos is not shifted.
        max_seq_len:
            The possible maximum sequence length for positional encoding
        use_after_mask:
            True value makes transformer unidirectional
        use_src_key_padding_mask:
            Padding simbols aren't used in attention bases on sequences lenghts
        use_norm_layer:
            Use or not LayerNorm
        is_reduce_sequence (bool):
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    Example:
    >>> model = TransformerEncoder(input_size=32)
    >>> x = PaddedBatch(torch.randn(10, 128, 32), torch.randint(20, 128, (10,)))
    >>> y = model(x)
    >>> assert y.payload.size() == (10, 128, 32)
    >>>
    >>> model = TransformerEncoder(input_size=32, is_reduce_sequence=True)
    >>> y = model(x)
    >>> assert y.size() == (10, 32)

    """
    def __init__(self,
                 input_size,
                 starter='randn',
                 shared_layers=False,
                 n_heads=8,
                 dim_hidden=256,
                 dropout=0.1,
                 n_layers=6,
                 use_positional_encoding=True,
                 use_start_random_shift=True,
                 max_seq_len=5000,
                 use_after_mask=False,
                 use_src_key_padding_mask=True,
                 use_norm_layer=True,
                 is_reduce_sequence=False,  # previous default behavior for TransformerSeqEncoder
                 ):
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.input_size = input_size
        self.shared_layers = shared_layers
        self.n_layers = n_layers
        self.use_after_mask = use_after_mask
        self.use_src_key_padding_mask = use_src_key_padding_mask
        self.use_positional_encoding = use_positional_encoding

        if starter == 'randn':
            self.starter = torch.nn.Parameter(torch.randn(1, 1, input_size), requires_grad=True)
        elif starter == 'zeros':
            self.starter = torch.nn.Parameter(torch.zeros(1, 1, input_size), requires_grad=False)
        else:
            raise AttributeError(f'Unknown train_starter: "{starter}". Expected one of [randn, zeros]')

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=n_heads,
            dim_feedforward=dim_hidden,
            dropout=dropout,
            batch_first=True,
        )
        enc_norm = torch.nn.LayerNorm(input_size) if use_norm_layer else None

        if self.shared_layers:
            self.enc_layer = enc_layer
            self.enc_norm = enc_norm
        else:
            self.enc = torch.nn.TransformerEncoder(enc_layer, n_layers, enc_norm)

        if self.use_positional_encoding:
            self.pe = PositionalEncoding(
                use_start_random_shift=use_start_random_shift,
                max_len=max_seq_len,
                d_model=input_size,
            )

    @staticmethod
    def generate_square_subsequent_mask(sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask[0, :] = 0.0
        mask[:, 0] = 0.0
        return mask

    def forward(self, x: PaddedBatch):
        B, T, H = x.payload.size()

        if self.use_after_mask:
            src_mask = self.generate_square_subsequent_mask(T + 1).to(x.device)
        else:
            src_mask = None

        if self.use_src_key_padding_mask:
            src_key_padding_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.long, device=x.device),
                (1 - x.seq_len_mask),
            ], dim=1).bool()
        else:
            src_key_padding_mask = None

        x_in = x.payload
        if self.use_positional_encoding:
            x_in = self.pe(x_in)
        x_in = torch.cat([self.starter.expand(B, 1, H), x_in], dim=1)

        if self.shared_layers:
            out = x_in
            for _ in range(self.n_layers):
                out = self.enc_layer(out, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
                if self.enc_norm is not None:
                    out = self.enc_norm(out)
        else:
            out = self.enc(x_in, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        if self.is_reduce_sequence:
            return out[:, 0, :]

        return PaddedBatch(out[:, 1:, :], x.seq_lens)

    @property
    def embedding_size(self):
        return self.input_size
