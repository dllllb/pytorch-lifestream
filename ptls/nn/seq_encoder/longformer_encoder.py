import random
import torch
from transformers import LongformerConfig, LongformerModel

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder


class LongformerEncoder(AbsSeqEncoder):
    """Used huggingface implementation of transformer
    Based on `transformers.LongformerModel`
    [Link](https://huggingface.co/docs/transformers/main/en/model_doc/longformer#transformers.LongformerModel)

    Transformer-based models are unable to process long sequences due to their self-attention operation,
    which scales quadratically with the sequence length. To address this limitation, was introduce the Longformer
    with an attention mechanism that scales linearly with sequence length, making it easy to process documents
    of thousands of tokens or longer. Longformer’s attention mechanism is a drop-in replacement for
    the standard self-attention and combines a local windowed attention with a task motivated global attention.

    Parameters
        input_size:
            input embedding size.
            Equals intermediate and output layer size cause transformer don't change vector dimentions
        num_attention_heads:
            The number of heads in the multiheadattention models
        intermediate_size:
            The dimension of the feedforward network model
        num_hidden_layers:
            The number of sub-encoder-layers in the encoder
        attention_window:
            Size of an attention window around each token
        max_position_embeddings:
            The possible maximum sequence length for positional encoding
        use_positional_encoding (bool):
            Use or not positional encoding
        use_start_random_shift (bool):
            True - starting pos of positional encoding randomly shifted when training
            This allow to train transformer with all range of positional encoding values
            False - starting pos is not shifted.
        is_reduce_sequence (bool):
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    Example:
    >>> model = LongformerEncoder(input_size=32)
    >>> x = PaddedBatch(torch.randn(10, 128, 32), torch.randint(20, 128, (10,)))
    >>> y = model(x)
    >>> assert y.payload.size() == (10, 128, 32)
    >>>
    >>> model = LongformerEncoder(input_size=32, is_reduce_sequence=True)
    >>> y = model(x)
    >>> assert y.size() == (10, 32)

    """
    def __init__(self,
                 input_size,
                 num_attention_heads: int = 1,
                 intermediate_size: int = 128,
                 num_hidden_layers: int = 1,
                 attention_window: int = 16,
                 max_position_embeddings=5000,
                 use_positional_encoding=True,
                 use_start_random_shift=True,
                 is_reduce_sequence=False,
                 ):
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.hidden_size = input_size
        self.max_position_embeddings = max_position_embeddings
        self.use_positional_encoding = use_positional_encoding
        self.use_start_random_shift = use_start_random_shift

        self.token_cls = torch.nn.Parameter(torch.randn(1, 1, input_size), requires_grad=True)

        self.transf = LongformerModel(
            config=LongformerConfig(
                hidden_size=input_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                vocab_size=4,
                max_position_embeddings=max_position_embeddings,
                attention_window=attention_window,
            ),
            add_pooling_layer=False,
        )

    def forward(self, x: PaddedBatch):
        B, T, H = x.payload.size()
        device = x.device

        if self.training and self.use_start_random_shift:
            start_pos = random.randint(0, self.max_position_embeddings - T - 1)
        else:
            start_pos = 0

        inputs_embeds = torch.cat([
            self.token_cls.expand(B, 1, H),
            x.payload,
        ], dim=1)
        attention_mask = torch.cat([
            torch.ones(B, 1, device=device),
            x.seq_len_mask.float(),
        ], dim=1)
        if self.use_positional_encoding:
            position_ids = torch.arange(T, device=device, dtype=torch.long).view(1, -1).expand(B, T) + 1 + start_pos
        else:
            position_ids = torch.zeros(B, T, device=device, dtype=torch.long)
        position_ids = torch.cat([
            torch.zeros(B, 1, device=device, dtype=torch.long),
            position_ids,
        ], dim=1)
        global_attention_mask = torch.cat([
            torch.ones(B, 1, device=device, dtype=torch.long),
            torch.zeros(B, T, device=device, dtype=torch.long),
        ], dim=1)

        out = self.transf(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            global_attention_mask=global_attention_mask,
        ).last_hidden_state

        if self.is_reduce_sequence:
            return out[:, 0, :]

        return PaddedBatch(out[:, 1:, :], x.seq_lens)

    @property
    def embedding_size(self):
        return self.hidden_size
