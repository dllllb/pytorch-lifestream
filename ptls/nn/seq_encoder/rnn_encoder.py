import torch
from torch import nn as nn

from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_step import LastStepEncoder
from ptls.data_load.padded_batch import PaddedBatch


# TODO: split it on GRU Rnn Encoder and LSTM Rnn Encoder
class RnnEncoder(AbsSeqEncoder):
    """Use torch recurrent layer network
    Based on `torch.nn.GRU` and `torch.nn.LSTM`

    Parameters
        input_size:
            input embedding size
        hidden_size:
            intermediate and output layer size
        type:
            'gru' or 'lstm'
            Type of rnn network
        bidir:
            Not implemented. Use default value for this parameter
        trainable_starter:
            'static' - use random learnable vector for rnn starter
            other values - use None as starter
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    Example:
    >>> model = RnnEncoder(
    >>>     input_size=5,
    >>>     hidden_size=6,
    >>>     is_reduce_sequence=False,
    >>> )
    >>> x = PaddedBatch(
    >>>     payload=torch.arange(4*5*8).view(4, 8, 5).float(),
    >>>     length=torch.tensor([4, 2, 6, 8])
    >>> )
    >>> out = model(x)
    >>> assert out.payload.shape == (4, 8, 6)

    """
    def __init__(self,
                 input_size=None,
                 hidden_size=None,
                 type='gru',
                 bidir=False,
                 trainable_starter='static',
                 is_reduce_sequence=False,  # previous default behavior RnnEncoder
                 ):
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.hidden_size = hidden_size
        self.rnn_type = type
        self.bidirectional = bidir
        if self.bidirectional:
            raise AttributeError('bidirectional RNN is not supported yet')
        self.trainable_starter = trainable_starter

        # initialize RNN
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size,
                self.hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=self.bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size,
                self.hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=self.bidirectional)
        else:
            raise Exception(f'wrong rnn type "{self.rnn_type}"')

        self.full_hidden_size = self.hidden_size if not self.bidirectional else self.hidden_size * 2

        # initialize starter position if needed
        if self.trainable_starter == 'static':
            num_dir = 2 if self.bidirectional else 1
            self.starter_h = nn.Parameter(torch.randn(num_dir, 1, self.hidden_size))

        self.reducer = LastStepEncoder()

    def forward(self, x: PaddedBatch, h_0: torch.Tensor = None):
        """

        :param x:
        :param h_0: None or [1, B, H] float tensor
                    0.0 values in all components of hidden state of specific client means no-previous state and
                    use starter for this client
                    h_0 = None means no-previous state for all clients in batch
        :return:
        """
        shape = x.payload.size()
        assert shape[1] > 0, "Batch can'not have 0 transactions"

        # prepare initial state
        if self.trainable_starter == 'static':
            starter_h = self.starter_h.expand(-1, shape[0], -1).contiguous()
            if h_0 is None:
                h_0 = starter_h
            elif h_0 is not None and not self.training:
                h_0 = torch.where(
                    (h_0.squeeze(0).abs().sum(dim=1) == 0.0).unsqueeze(0).unsqueeze(2).expand(*starter_h.size()),
                    starter_h,
                    h_0,
                )
            else:
                raise NotImplementedError('Unsupported mode: cannot mix fixed X and learning Starter')

        # pass-through rnn
        if self.rnn_type == 'lstm':
            out, _ = self.rnn(x.payload)
        elif self.rnn_type == 'gru':
            out, _ = self.rnn(x.payload, h_0)
        else:
            raise Exception(f'wrong rnn type "{self.rnn_type}"')

        out = PaddedBatch(out, x.seq_lens)
        if self.is_reduce_sequence:
            return self.reducer(out)
        return out

    @property
    def embedding_size(self):
        return self.hidden_size
