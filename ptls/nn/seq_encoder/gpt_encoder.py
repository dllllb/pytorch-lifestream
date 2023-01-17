import random
import torch
from transformers import GPT2Config, GPT2Model

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_step import LastStepEncoder

class GptEncoder(AbsSeqEncoder):
    """Used huggingface implementation of GPT decoder
    Based on `transformers.GPT2Model`

    Parameters
    ----------
        n_embd:
            input embedding size.
            Equals intermediate and output layer size cause transformer don't change vector dimentions
        n_head:
            The number of heads in the multiheadattention models
        n_inner:
            The dimension of the feedforward network model
        n_layer:
            The number of sub-decoder-layers in the decoder
        activation_function:
            Activation function is used.
        n_positions:
            The possible maximum sequence length for positional encoding
        resid_pdrop:
            Dropout probability of residual connections of decoder layers.
        embd_pdrop:
            Dropout probability of embeddings.
        attn_pdrop:
            Dropout probability of attention matrix.
        use_positional_encoding (bool):
            Use or not positional encoding
        use_start_random_shift (bool):
            True - starting pos of positional encoding randomly shifted when training
            This allow to train transformer with all range of positional encoding values
            False - starting pos is not shifted.
        is_reduce_sequence (bool):
            False - returns PaddedBatch with all transactions embeddings
            True - returns last embedding of sequence 
    """

    def __init__(self,
                 n_embd=768,
                 n_layer=12,
                 n_head=12,
                 n_inner=2048,
                 activation_function="gelu_new",
                 resid_pdrop=0.1,
                 embd_pdrop=0.1,
                 attn_pdrop=0.1,
                 n_positions=1024,
                 use_positional_encoding=True,
                 use_start_random_shift=True,
                 is_reduce_sequence=False,
                 ):
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.n_embd = n_embd
        self.n_positions = n_positions
        self.use_positional_encoding = use_positional_encoding
        self.use_start_random_shift = use_start_random_shift
        
        if is_reduce_sequence:
            self.last_step = LastStepEncoder()
        
        self.transf = GPT2Model(
            config=GPT2Config(
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_inner=n_inner,
                activation_function=activation_function,
                resid_pdrop=resid_pdrop,
                embd_pdrop=embd_pdrop,
                attn_pdrop=attn_pdrop,
                n_positions=n_positions,
                vocab_size=4,
            ),
        )
        
    def forward(self, x: PaddedBatch):
        B, T, H = x.payload.size()
        device = x.device

        if self.training and self.use_start_random_shift:
            start_pos = random.randint(0, self.n_positions - T - 1)
        else:
            start_pos = 0

        inputs_embeds = x.payload
        attention_mask = x.seq_len_mask.float()

        if self.use_positional_encoding:
            position_ids = torch.arange(T, device=device, dtype=torch.long).view(1, -1).expand(B, T) + start_pos
        else:
            position_ids = torch.zeros(B, T, device=device, dtype=torch.long)

        out = self.transf(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state

        out = PaddedBatch(out, x.seq_lens)

        if self.is_reduce_sequence:
            return self.last_step(out)
        return out

    @property
    def embedding_size(self):
        return self.n_embd
