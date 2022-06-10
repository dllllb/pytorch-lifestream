import torch
from typing import List, Dict
from omegaconf import OmegaConf

from ptls.seq_encoder.agg_feature_model import AggFeatureSeqEncoder
from ptls.seq_encoder.rnn_encoder import RnnSeqEncoder, RnnSeqEncoderDistributionTarget
from ptls.seq_encoder.transf_seq_encoder import TransfSeqEncoder
from ptls.seq_encoder.statistics_encoder import StatisticsEncoder
from ptls.seq_encoder.dummy_encoder import DummyEncoder
from ptls.trx_encoder import TrxEncoder


class SequenceEncoder(torch.nn.Module):
    r"""Deprecated. Use `ptls.seq_encoder.abs_seq_encoder.AbsSeqEncoder` implementations

    Parameters
    ----------
     category_features: Dict[str, int],
     numeric_features: List[str],
     trx_embedding_size: int. Default: 16
        The number of features in the category fields embedding.
     trx_embedding_noize: float. Default: 0.0
        When > 0 applies additive Gaussian noise to category fields embeddings.
     encoder_type: str. Default: 'rnn'
       Type of encoder. Options: 'rnn' for RNN, 'transformer' for transformer, and 'agg_features' for
       the net that aggregated statistic from the sequence
     rnn_hidden_size: int. Default: 512
        The number of features in the RNN hidden state. Used only if encoder_type == 'rnn'.
     rnn_type: str. Default: 'gru'
        Type of RNN. Options: 'gru', 'lstm'. Used only if encoder_type == 'rnn'.
     rnn_trainable_starter: bool. Default: False
        Whether to use trainable starter for the RNN. Used only if encoder_type == 'rnn'.
     rnn_bidirectional: bool. Default: False
         If True, becomes a bidirectional RNN. Used only if encoder_type == 'rnn'.
     transformer_input_size: int. Default: 512
        Transformer input size, used only if encoder_type == 'transformer'.
     transformer_dim_hidden: int. Default: 256
        The number of features in the attention hidden state. Used only if encoder_type == 'transformer'.
     transformer_n_layers: int. Default: 4
        The number of layers in transformer. Used only if encoder_type == 'transformer'.
     transformer_n_heads: int. Default: 4
        The number of heads in each transformer attention layer. Used only if encoder_type == 'transformer'.
     transformer_shared_layers: bool. Default: False
        Whether to share weights in transformer layers on not. Used only if encoder_type == 'transformer'.
     transformer_use_after_mask: bool. Default: False
         Whether to share weights in transformer layers on not. Used only if encoder_type == 'transformer'.
     transformer_use_src_key_padding_mask: bool. Default: False
         Whether to use padding mask on not. Used only if encoder_type == 'transformer'.
     transformer_use_positional_encoding: bool. Default: False
         Whether to use positional encoding on not. Used only if encoder_type == 'transformer'.
     transformer_train_starter: bool. Default: False
        Whether to use trainable starter for the transformer. Used only if encoder_type == 'transformer'.
     transformer_dropout: float. Default: 0.0
        The dropout after attention layer value. Used only if encoder_type == 'transformer'.

     """

    def __init__(self,
                 category_features: Dict[str, int],
                 numeric_features: List[str],
                 trx_embedding_size: int = 16,
                 trx_embedding_noise: float = 0.0,
                 trx_norm_embeddings: bool = False,
                 trx_use_batch_norm_with_lens: bool = False,
                 trx_clip_replace_value: bool = False,
                 was_logified: bool = True,
                 log_scale_factor: float = 1.0,
                 encoder_type: str = 'rnn',
                 rnn_hidden_size: int = 512,
                 rnn_type: str = 'gru',
                 rnn_trainable_starter: str = None,
                 rnn_bidirectional: bool = False,
                 transformer_input_size: int = 512,
                 transformer_dim_hidden: int = 256,
                 transformer_n_layers: int = 4,
                 transformer_n_heads: int = 4,
                 transformer_shared_layers: bool = False,
                 transformer_use_after_mask: bool = False,
                 transformer_use_src_key_padding_mask: bool = False,
                 transformer_use_positional_encoding: bool = False,
                 transformer_train_starter: bool = False,
                 transformer_dropout: float = 0.0,
                 transformer_max_seq_len: int = 1200):

        super().__init__()

        embeddings = {k: {'in': v, 'out': trx_embedding_size} for k, v in category_features.items()}
        numeric_values = {k: 'identity' for k in numeric_features}

        trx_encoder = TrxEncoder(embeddings,
                                 numeric_values,
                                 trx_embedding_noise,
                                 trx_norm_embeddings,
                                 trx_use_batch_norm_with_lens,
                                 trx_clip_replace_value)

        if encoder_type == 'rnn':
            model = RnnSeqEncoder(trx_encoder,
                                  None,
                                  rnn_hidden_size,
                                  rnn_type,
                                  rnn_bidirectional,
                                  rnn_trainable_starter)

        elif encoder_type == 'transformer':
            model = TransfSeqEncoder(transformer_input_size,
                                     transformer_train_starter,
                                     transformer_shared_layers,
                                     transformer_n_heads,
                                     transformer_dim_hidden,
                                     transformer_dropout,
                                     transformer_n_layers,
                                     transformer_use_positional_encoding,
                                     transformer_max_seq_len,
                                     transformer_use_after_mask,
                                     transformer_use_src_key_padding_mask)

        elif encoder_type == 'agg_features':
            model = AggFeatureSeqEncoder(embeddings,
                                         numeric_values,
                                         was_logified,
                                         log_scale_factor)

        else:
            raise AttributeError(f'Unknown encoder_type: {encoder_type}')

        self.model = model
        # self.params = params

    @property
    def is_reduce_sequence(self):
        return self.model._is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self.model._is_reduce_sequence = value

    @property
    def category_max_size(self):
        return self.model.category_max_size

    @property
    def category_names(self):
        return self.model.category_names

    @property
    def embedding_size(self):
        return self.model.embedding_size

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.model(x)
