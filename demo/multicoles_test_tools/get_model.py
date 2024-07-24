import torch
import ptls.frames
import ptls.frames.coles
from ptls.nn.normalization import L2NormEncoder
from ptls.frames.coles.losses import ContrastiveLoss, CLUBLoss
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.frames.coles.metric import BatchRecallTopK
from functools import partial


class ResNet(torch.nn.Module):
    def __init__(self, h, use_layernorm=True, dropout=0):
        super().__init__()
        layers = list()
        for _ in range(2):
            layers.append(torch.nn.Linear(h, h))
            if use_layernorm:
                layers.append(torch.nn.LayerNorm(h))
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, inp):
        return self.net(inp) + inp


class ClfDisc(torch.nn.Module):
    def __init__(self, inp1=400, inp2=400, h=512, n_res_blocks=3, use_bn=True, use_l2_norm=True):
        super().__init__()
        layers_a = [torch.nn.Linear(inp1, h), torch.nn.ReLU()]
        layers_a.extend([ResNet(h) for _ in range(n_res_blocks)])
        if use_bn:
            layers_a.append(torch.nn.BatchNorm1d(h, affine=False))
        if use_l2_norm:
            layers_a.append(L2NormEncoder())
        self.a = torch.nn.Sequential(*layers_a)

        layers_b = [torch.nn.Linear(inp2, h), torch.nn.ReLU()]
        layers_b.extend([ResNet(h) for _ in range(n_res_blocks)])
        if use_bn:
            layers_b.append(torch.nn.BatchNorm1d(h, affine=False))
        if use_l2_norm:
            layers_b.append(L2NormEncoder())
        self.b = torch.nn.Sequential(*layers_b)

    def forward(self, domain_a, domain_b):
        a = self.a(domain_a)
        b = self.b(domain_b)
        return -(((a - b) ** 2).sum(axis=-1, keepdims=True))


class IdentityDisc(torch.nn.Module):
    def __init__(self, inp=400, h=512):
        super().__init__()
        self.inp = inp
        self.h = h
        self.k = self.h / self.inp
        self.register_parameter('lame', torch.nn.Parameter(torch.tensor(0.)))

    def forward(self, domain_a, domain_b):
        return -((domain_a - domain_b)**2).sum(axis=-1, keepdims=True) * self.k + 0 * self.lame


def get_coles_module(trx_conf, input_size, hsize):
    pl_module = ptls.frames.coles.CoLESModule(
        validation_metric=BatchRecallTopK(K=4, metric='cosine'),
        seq_encoder=ptls.nn.RnnSeqEncoder(
            trx_encoder=ptls.nn.TrxEncoder(**trx_conf),
            input_size=input_size,
            type='gru',
            hidden_size=hsize,
            is_reduce_sequence=True
        ),
        head=ptls.nn.Head(use_norm_encoder=True),
        loss=ContrastiveLoss(
            margin=1.,
            sampling_strategy=HardNegativePairSelector(neg_count=5),
        ),
        optimizer_partial=partial(torch.optim.Adam, lr=0.001, weight_decay=0.0),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9025)
    )
    return pl_module


def get_static_multicoles_module(trx_conf, input_size, embed_coef, hsize, clf_hsize, first_model_name):
    coles_loss = ContrastiveLoss(margin=1., sampling_strategy=HardNegativePairSelector(neg_count=5))
    club_loss = CLUBLoss(log_var=-2, emb_coef=1, prob_coef=1.)
    discriminator_model = ClfDisc(inp1=hsize, inp2=hsize, h=clf_hsize)

    seq_encoder = ptls.nn.RnnSeqEncoder(
        trx_encoder=ptls.nn.TrxEncoder(**trx_conf),
        input_size=input_size,
        type='gru',
        hidden_size=hsize,
        is_reduce_sequence=True
    )

    pl_module = ptls.frames.coles.MultiCoLESModule(
        head=ptls.nn.Head(use_norm_encoder=True),
        validation_metric=BatchRecallTopK(K=4, metric='cosine'),
        loss=coles_loss,
        discriminator_loss=club_loss,
        seq_encoder=seq_encoder,
        discriminator=discriminator_model,
        optimizer_partial=partial(torch.optim.Adam, lr=0.001, weight_decay=0.0),
        d_optimizer_partial=partial(torch.optim.Adam, lr=0.001),
        trained_encoders=[first_model_name],
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9025),
        coles_coef=1.,
        embed_coef=embed_coef
    )
    return pl_module
