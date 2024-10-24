import torch
import warnings
from torch import nn as nn

from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_step import LastStepEncoder, LastMaxAvgEncoder, FirstStepEncoder
from ptls.data_load.padded_batch import PaddedBatch

REDUCE_DICT = {'last_step': LastStepEncoder,
               'first_step': LastStepEncoder,
               'last_max_avg': LastMaxAvgEncoder}

RNN_MODEL_DICT = {'lstm': nn.LSTM,
                  'gru': nn.GRU}


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
            Bidirectional RNN
        dropout:
            RNN dropout
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
                 num_layers=1,
                 dropout=0,
                 trainable_starter='static',
                 is_reduce_sequence=False,  # previous default behavior RnnEncoder
                 reducer='last_step'
                 ):
        self.bidirectional = bidir
        self.trainable_starter = trainable_starter
        self.hidden_size = hidden_size
        self.rnn_type = type
        self.num_layers = num_layers
        self.reducer_name = reducer
        self.full_hidden_size = self.hidden_size if not self.bidirectional else self.hidden_size * 2
        self.model_params = dict(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                 batch_first=True,
                                 bidirectional=self.bidirectional, dropout=dropout)

        super().__init__(is_reduce_sequence=is_reduce_sequence)
        if self.bidirectional:
            warnings.warn("Backward direction in bidir RNN takes into account paddings at the end of sequences!")
        # initialize starter position if needed
        if self.trainable_starter == 'static':
            num_dir = 2 if self.bidirectional else 1
            self.starter_h = nn.Parameter(torch.randn(self.num_layers * num_dir, 1, self.hidden_size))

        self.reducer = self.get_reducer
        self.rnn = RNN_MODEL_DICT[self.rnn_type](**self.model_params)

    @property
    def get_reducer(self):
        return REDUCE_DICT[self.reducer_name]()

    @property
    def embedding_size(self):
        return self.hidden_size

    def __create_default_init_state(self, shape):
        num_dir = 2 if self.bidirectional else 1
        return torch.tanh(self.starter_h.expand(self.num_layers * num_dir, shape[0], -1).contiguous())

    def __update_init_state(self, h_0, shape):
        return torch.where((h_0.squeeze(0).abs().sum(dim=1) == 0.0)
                           .unsqueeze(0).unsqueeze(2).expand(*self.__create_default_init_state(shape).size()),
                           *self.__create_default_init_state(shape), h_0, )

    def _eval_init_state(self, shape, h_0):
        # prepare initial state
        if not self.trainable_starter == 'static':
            return None
        else:
            have_init_state_for_predict = all([h_0 is not None, not self.training])
            h_0 = self.__create_default_init_state(shape) if h_0 is None else self.__update_init_state(h_0, shape)
            return h_0

    def forward(self, x: PaddedBatch, h_0: torch.Tensor = None):
        """

        :param x:
        :param h_0: None or [1, B, H] float tensor
                    0.0 values in all components of hidden state of specific client means no-previous state and
                    use starter for this client
                    h_0 = None means no-previous state for all clients in batch
        :return:
        """

        assert x.payload.size()[1] > 0, "Batch can'not have 0 transactions"
        h_0 = self._eval_init_state(x.payload.size(), h_0)
        out, _ = self.rnn(x.payload, h_0) if self.rnn_type == 'gru' else self.rnn(x.payload)
        out = PaddedBatch(out, x.seq_lens)
        return self.reducer(out) if self.is_reduce_sequence else out
