import random
import torch
from transformers import GPT2Config, GPT2Model

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder


class GptEncoder(AbsSeqEncoder):
    
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

        self.token_cls = torch.nn.Parameter(torch.randn(1, 1, n_embd), requires_grad=True)
        
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

        return out

        # if self.is_reduce_sequence:
        #     return out[:, -1, :]
        # else:
        #     return PaddedBatch(out, x.seq_lens)

    @property
    def embedding_size(self):
        return self.n_embd
