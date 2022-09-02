import torch
from torch import nn as nn


class NoisyEmbedding(nn.Embedding):
    """
    Embeddings with additive gaussian noise with mean=0 and user-defined variance.
    *args and **kwargs defined by usual Embeddings
    Args:
        noise_scale (float): when > 0 applies additive noise to embeddings.
            When = 0, forward is equivalent to usual embeddings.
        dropout (float): probability of embedding axis to be dropped. 0 means no dropout at all.
        spatial_dropout (bool): whether to dropout full dimension of embedding in the whole sequence.

    For other parameters defenition look at nn.Embedding help
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                 noise_scale=0, dropout=0, spatial_dropout=False):
        if max_norm is not None:
            raise AttributeError("Please don't use embedding normalisation. "
                                 "https://github.com/pytorch/pytorch/issues/44792")

        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight)
        self.noise = torch.distributions.Normal(0, noise_scale) if noise_scale > 0 else None
        self.scale = noise_scale
        self.spatial_dropout = spatial_dropout
        self.dropout = nn.Dropout2d(dropout) if spatial_dropout else nn.Dropout(dropout)

        # This is workaround for https://github.com/pytorch/pytorch/issues/44792
        # Weight are normalized after forward pass
        _ = super().forward(torch.arange(num_embeddings))

    def forward(self, x):
        x = super().forward(x)
        if self.spatial_dropout:
            x = self.dropout(x.permute(0, 2, 1).unsqueeze(3)).squeeze(3).permute(0, 2, 1)
        else:
            x = self.dropout(x)
        if self.training and self.scale > 0:
            x += self.noise.sample((self.weight.shape[1], )).to(self.weight.device)
        return x
