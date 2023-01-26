import torch
from torch.nn import MultiheadAttention
from torch.nn.functional import relu, gelu

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder


class Encoder(AbsSeqEncoder):
    """Custom builded transformer encoder

    Parameters
    ----------
    input_size:
        input embedding size.
        Equals intermediate and output layer size cause transformer don't change vector dimentions
    num_attention_heads:
        The number of heads in the multiheadattention models
    intermediate_size:
        The dimension of the feedforward network model
    num_hidden_layers:
        The number of sub-encoder-layers in the encoder
    attn_block_mode:
        default - without training connection weights 
        rezero - from (https://arxiv.org/pdf/2003.04887.pdf)
    self_attn_mode:
        quadratic - classic torch.nn.MultiheadAttention
        linear-flow - FlowFormer from (https://arxiv.org/pdf/2203.16194.pdf)
    aggregation_mode:
        Aggregation strategies - {mean, sum, max}
    layer_norm:
        None - default value, without layer normalization
        pre - before attn and before feedforward
        post - after attn and after feedforward
    is_reduce_sequence (bool):
        False - returns PaddedBatch with all transactions embeddings after aggregation step
        True - returns Tensor with all transactions embeddings after aggregation step
    """

    def __init__(self, 
                input_size: int,
                intermediate_size: int = 2048, 
                num_hidden_layers: int = 8,
                num_attention_heads: int = 8,
                attn_block_mode: str = 'rezero', 
                self_attn_mode: str = 'quadratic', 
                aggregation_mode: str = 'mean',
                layer_norm=None,
                is_reduce_sequence=True):
        super().__init__(is_reduce_sequence=is_reduce_sequence)
        
        self.transformer = torch.nn.Sequential(
            *[AttentionBlock(
                input_size, intermediate_size, attn_block_mode, self_attn_mode, layer_norm, num_attention_heads
            ) for _ in range(num_hidden_layers)]
        )
        
        self.aggregation = Aggregation(reduction=aggregation_mode)
        self.is_reduce_sequence = is_reduce_sequence

    def forward(self, x: PaddedBatch):
        out = self.transformer(x.payload)
        out = self.aggregation(out)
        if self.is_reduce_sequence:
            return out
        return PaddedBatch(out, x.seq_lens)

    @property
    def embedding_size(self):
        return self.hidden_size


class FlowAttention(torch.nn.Module):
    """FlowFormer from (https://arxiv.org/pdf/2203.16194.pdf)"""

    def __init__(self, input_dim, num_heads):
        super().__init__()
        embed_dim = input_dim
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = torch.nn.Linear(input_dim, 3*embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)   # [Batch, SeqLen, Head, Dims]

        q = torch.sigmoid(q)
        k = torch.sigmoid(k)

        i = torch.sum(q * torch.sum(k, dim=1, keepdim=True), dim=3, keepdim=True)
        o = torch.sum(k * torch.sum(q, dim=1, keepdim=True), dim=3, keepdim=True)

        i_hat = torch.sum(q * torch.sum(k / o, dim=1, keepdim=True), dim=3, keepdim=True)
        o_hat = torch.sum(k * torch.sum(q / i, dim=1, keepdim=True), dim=3, keepdim=True)

        r = torch.matmul(q / i, torch.matmul(k.transpose(-2, -1), v * torch.softmax(o_hat, dim=1))) * torch.sigmoid(i_hat)

        values = r.reshape(batch_size, seq_length, embed_dim)
        out = self.out_proj(values)
        
        return out

class MLP(torch.nn.Module):
    
    def __init__(self, n_in, n_hidden, n_out, depth=2):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hidden),
            torch.nn.GELU(),
            *[torch.nn.Sequential(
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
            ) for _ in range(depth-1)],
            torch.nn.Linear(n_hidden, n_out)
        )
        
    def forward(self, X):
        return self.mlp(X)
    
class Attention(torch.nn.Module):
    
    def __init__(self, embed_dim, num_heads, self_attn):
        super().__init__()
        
        self.self_attn = self_attn
        if self_attn=="quadratic":
            self.attn = MultiheadAttention(embed_dim, num_heads, batch_first=True)
        elif self_attn=="linear-flow":
            self.attn = FlowAttention(embed_dim, num_heads)
        elif self_attn=="linear-cross":
            pass # TODO
        
    def forward(self, X):
        
        if self.self_attn=="quadratic":
            X, _ = self.attn(X, X, X)
        elif self.self_attn=="linear-flow":
            X = self.attn(X)
        elif self.self_attn=="linear-cross":
            pass
        
        return X
    
class AttentionBlock(torch.nn.Module):
    
    def __init__(self, embed_dim, mlp_hidden_dim, attn_block, self_attn, layer_norm, num_heads=4):
        super().__init__()
        
        self.attn_block = attn_block
        self.layer_norm = layer_norm
        
        if self.attn_block=="rezero":
            self.alpha_attn = torch.nn.Parameter(torch.normal(torch.tensor(0.), torch.tensor(1e-6))) 
            self.alpha_mlp = torch.nn.Parameter(torch.normal(torch.tensor(0.), torch.tensor(1e-6)))
        
        self.attn = Attention(embed_dim, num_heads, self_attn)
        self.linear1 = torch.nn.Linear(embed_dim, mlp_hidden_dim)
        self.linear2 = torch.nn.Linear(mlp_hidden_dim, embed_dim)
        
    def forward(self, X):
        
        if self.attn_block=="rezero":
            if self.layer_norm == 'pre':
                X = X / (X.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
            Z = X + self.alpha_attn * self.attn(X)
            if self.layer_norm:
                Z = Z / (Z.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
            Z = Z + self.alpha_mlp * self.linear2(gelu(self.linear1(Z)))
            if self.layer_norm == 'post':
                Z = Z / (Z.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
            X = X + Z # double residual
        
        else: # no-ln
            if self.layer_norm == 'pre':
                X = X / (X.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
            Z = X + self.attn(X)
            if self.layer_norm:
                Z = Z / (Z.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
            Z = Z + self.linear2(gelu(self.linear1(Z)))
            if self.layer_norm == 'post':
                Z = Z / (Z.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
            X = X + Z # double residual
    
        return X
    
class Aggregation(torch.nn.Module):
    
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, X):
        
        if self.reduction=="mean":
            x = X.mean(dim=1)
        elif self.reduction=="sum":
            x = X.sum(dim=1)
        elif self.reduction=="max":
            x, _ = torch.max(x, dim=1)
        else:
            x = X[:, 0]
        return x