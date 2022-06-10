import torch

from ptls.seq_encoder.rnn_encoder import RnnEncoder
from ptls.seq_encoder.transformer_encoder import TransformerEncoder


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
                 ):
        super().__init__()

        self.trx_encoder = trx_encoder
        self.seq_encoder = seq_encoder_cls(
            input_size=input_size if input_size is not None else trx_encoder.output_size,
            is_reduce_sequence=is_reduce_sequence,
            **seq_encoder_params,
        )

    @property
    def is_reduce_sequence(self):
        return self.seq_encoder.is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self.seq_encoder.is_reduce_sequence = value

    @property
    def category_max_size(self):
        raise self.trx_encoder.category_max_size

    @property
    def category_names(self):
        raise self.trx_encoder.category_names

    @property
    def embedding_size(self):
        return self.seq_encoder.embedding_size

    def forward(self, x):
        x = self.trx_encoder(x)
        x = self.seq_encoder(x)
        return x


class RnnSeqEncoder(SeqEncoderContainer):
    """SeqEncoderContainer with RnnEncoder

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
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=RnnEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )

    def forward(self, x, h_0=None):
        x = self.trx_encoder(x)
        x = self.seq_encoder(x, h_0)
        return x


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
