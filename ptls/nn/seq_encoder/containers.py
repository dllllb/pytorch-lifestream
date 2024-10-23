import torch
from copy import deepcopy

from ptls.nn.seq_encoder.rnn_encoder import RnnEncoder
from ptls.nn.seq_encoder.transformer_encoder import TransformerEncoder
from ptls.nn.seq_encoder.longformer_encoder import LongformerEncoder
from ptls.nn.seq_encoder.custom_encoder import Encoder
from ptls.nn.seq_encoder.utils import reset_parameters


class SeqEncoderContainer(torch.nn.Module):
    """Base container class for Sequence encoder.
    Include `TrxEncoder` and `ptls.seq_encoder.abs_seq_encoder.AbsSeqEncoder` implementation

    Parameters
        trx_encoder:
            TrxEncoder object
        seq_encoder_cls:
            AbsSeqEncoder implementation class
        input_size:
            input_size parameter for seq_encoder_cls
            If None: input_size = trx_encoder.output_size
            Set input_size explicit or use None if your trx_encoder object has output_size attribute
        seq_encoder_params:
            dict with params for seq_encoder_cls initialisation
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    """
    def __init__(self,
                 trx_encoder,
                 seq_encoder_cls,
                 input_size,
                 seq_encoder_params,
                 is_reduce_sequence=True,  # most frequent default behavior
                 n_encoders=1,
                 ):
        super().__init__()

        self.n_encoders = n_encoders
        if self.n_encoders == 1:
            self.trx_encoder = trx_encoder
            self.seq_encoder = seq_encoder_cls(
                input_size=input_size if input_size is not None else trx_encoder.output_size,
                is_reduce_sequence=is_reduce_sequence,
                **seq_encoder_params,
            )
        elif self.n_encoders > 1:
            seq_encoder = seq_encoder_cls(
                input_size=input_size if input_size is not None else trx_encoder.output_size,
                is_reduce_sequence=is_reduce_sequence,
                **seq_encoder_params,
            )
            self.trx_encoder = torch.nn.ModuleList([deepcopy(trx_encoder) for _ in range(self.n_encoders)])
            self.seq_encoder = torch.nn.ModuleList([deepcopy(seq_encoder) for _ in range(self.n_encoders)])
            for i in range(self.n_encoders):
                reset_parameters(self.trx_encoder[i])
                reset_parameters(self.seq_encoder[i])
        else:
            raise NotImplemented

    @property
    def is_reduce_sequence(self):
        return self.seq_encoder.is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self.seq_encoder.is_reduce_sequence = value

    @property
    def category_max_size(self):
        return self.trx_encoder.category_max_size

    @property
    def category_names(self):
        return self.trx_encoder.category_names

    @property
    def embedding_size(self):
        return self.seq_encoder.embedding_size

    def forward(self, x):
        if self.n_encoders == 1:
            x = self.trx_encoder(x)
            y_h = self.seq_encoder(x)
        elif self.n_encoders > 1:
            y_h = list()
            for trx, seq in zip(self.trx_encoder, self.seq_encoder):
                y_h.append(seq(trx(x)))
        else:
            raise NotImplemented
        return y_h


class RnnSeqEncoder(SeqEncoderContainer):
    """SeqEncoderContainer with RnnEncoder

    Supports incremental embedding calculation.
    Each RNN step requires previous hidden state. Hidden state passed through the iterations during sequence processing.
    Starting hidden state required by RNN. Starting hidden state are depends on `RnnEncoder.trainable_starter`.
    You can also provide starting hidden state to `forward` method as `h_0`.
    This can be useful when you need to `update` your embedding with new transactions.

    Example:
        >>> seq_encoder = RnnSeqEncoder(...)
        >>> embedding_0 = seq_encoder(data_0)
        >>> embedding_1 = seq_encoder(data_1, h_0=embedding_0)
        >>> embedding_2a = seq_encoder(data_2, h_0=embedding_1)
        >>> embedding_2b = seq_encoder(data_2)
        >>> embedding_2c = seq_encoder(data_0 + data_1 + data_2)

    `embedding_2a` takes into account all transactions from `data_0`, `data_1` and `data_2`.
    `embedding_2b` takes into account only transactions from `data_2`.
    `embedding_2c` is the same as `embedding_2a`.
    `embedding_2a` calculated faster than `embedding_2c`.

    Incremental calculation works fast when you have long sequences and short updates. RNN just process short update.


    Parameters
        trx_encoder:
            TrxEncoder object
        input_size:
            input_size parameter for RnnEncoder
            If None: input_size = trx_encoder.output_size
            Set input_size explicit or use None if your trx_encoder object has output_size attribute
        **seq_encoder_params:
            params for RnnEncoder initialisation
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    """
    def __init__(self,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,
                 n_encoders=1,
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=RnnEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
            n_encoders=n_encoders,
        )

    def forward(self, x, h_0=None):
        if self.n_encoders == 1:
            x = self.trx_encoder(x)
            y_h = self.seq_encoder(x, h_0)
        elif self.n_encoders > 1:
            y_h = list()
            for trx, seq in zip(self.trx_encoder, self.seq_encoder):
                x_ = trx(x)
                y_ = seq(x_, h_0)
                y_h.append(y_)
        else:
            raise NotImplemented
        return y_h


class TransformerSeqEncoder(SeqEncoderContainer):
    """SeqEncoderContainer with TransformerEncoder

    Parameters
        trx_encoder:
            TrxEncoder object
        input_size:
            input_size parameter for TransformerEncoder
            If None: input_size = trx_encoder.output_size
            Set input_size explicit or use None if your trx_encoder object has output_size attribute
        **seq_encoder_params:
            params for TransformerEncoder initialisation
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    """

    def __init__(self,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=TransformerEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )


class LongformerSeqEncoder(SeqEncoderContainer):
    """SeqEncoderContainer with TransformerEncoder

    Parameters
        trx_encoder:
            TrxEncoder object
        input_size:
            input_size parameter for LongformerEncoder
            If None: input_size = trx_encoder.output_size
            Set input_size explicit or use None if your trx_encoder object has output_size attribute
        **seq_encoder_params:
            params for LongformerEncoder initialisation
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    """

    def __init__(self,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=LongformerEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )

class CustomSeqEncoder(SeqEncoderContainer):
    """SeqEncoderContainer with Custom builded transformer encoder

    Parameters
        trx_encoder:
            TrxEncoder object
        input_size:
            input_size parameter for Custom Encoder
            If None: input_size = trx_encoder.output_size
            Set input_size explicit or use None if your trx_encoder object has output_size attribute
        **seq_encoder_params:
            params for for Custom Encoder initialisation
        is_reduce_sequence (bool):
            False - returns PaddedBatch with all transactions embeddings after aggregation step
            True - returns Tensor with all transactions embeddings after aggregation step

    """
    def __init__(self,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=Encoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )

    def forward(self, x):
        z_trx = self.trx_encoder(x)
        out = self.seq_encoder(z_trx)
        return out