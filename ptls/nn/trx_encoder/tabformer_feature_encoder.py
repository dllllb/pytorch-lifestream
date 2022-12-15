import torch
import torch.nn as nn

class TabFormerFeatureEncoder(nn.Module):
    """TabFormerFeatureEncoder: encodes input batch of shape (B, T, F, E),
           where:
               B - batch size,
               T - sequence length,
               F - number of features
               E - embedding dimension for each feature
       and returns output batch of same shape.

       Encoding is performed as in [Tabular Transformers for Modeling Multivariate Time Series](https://arxiv.org/abs/2011.01843)

       Parameters
       ----------
       n_cols: number of features to encode,
       emb_dim: feature embedding dimension,
       n_heads: number of heads in transformer,
       n_layers: number of layers in transformer,
       out_hidden: out hidden dimension for each feature
    """

    def __init__(self, n_cols: int,
                       emb_dim: int,
                       transf_feedforward_dim: int = 64,
                       n_heads: int = 8,
                       n_layers: int = 1,
                       out_hidden: int = None):
        super().__init__()

        out_hidden = out_hidden if out_hidden else emb_dim * n_cols
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=transf_feedforward_dim, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lin_proj = nn.Linear(emb_dim * n_cols, out_hidden)

    def forward(self, input_embeds):
        embeds_shape = list(input_embeds.size())
        input_embeds = input_embeds.view([-1] + embeds_shape[-2:])  # (B, T, NUM_F, H) -> (B*T, NUM_F, H)
        out_embeds = self.transformer_encoder(input_embeds)
        out_embeds = out_embeds.contiguous().view(embeds_shape[0:2]+[-1])  # (B*T, NUM_F, H) -> (B, T, NUM_F*H)
        out_embeds = self.lin_proj(out_embeds)  # (B, T, H1) -> (B, T, H2)
        return out_embeds

