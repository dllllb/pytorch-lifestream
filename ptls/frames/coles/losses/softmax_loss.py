import torch
import torch.distributed as dist

from ptls.frames.coles.losses.dist_utils import all_gather_and_cat

class SoftmaxLoss(torch.nn.Module):
    """Also known as NCE loss
    All positive pairs as `anchor`-`pos_sample` concated with all possible `neg_samples` from batch.
    Softmax is applied ower `distances` between `anchor` ans pos or neg samples.
    `distances` is a dot product.
    Loss minimize `-log()` for `anchor-pos` position.
    
    Params:
        temperature:
            `softmax(distances / temperature)` - scale a sub-exponent expression.
            default 0.05 value is for l2-normalized `embeddings` where dot product distance is in range [-1, 1]
    """
    def __init__(self, temperature=0.05, distributed_mode = False):
        super().__init__()
        
        self.temperature = temperature
        self.distributed_mode = distributed_mode
        
    def forward(self, embeddings, classes):
        if dist.is_initialized() and self.distributed_mode:
            dist.barrier()
            embeddings = all_gather_and_cat(embeddings)
            classes = classes + (classes.max()+1) * dist.get_rank()
            classes = all_gather_and_cat(classes)
        d = torch.einsum('bh,kh->bk', embeddings, embeddings) / self.temperature
        
        ix_pos = classes.unsqueeze(1) == classes.unsqueeze(0)
        ix_pos.fill_diagonal_(0)
        ix_a, ix_pos = ix_pos.nonzero(as_tuple=True)
        
        _, ix_neg = (classes[ix_a].unsqueeze(1) != classes.unsqueeze(0)).nonzero(as_tuple=True)
        ix_all = torch.cat([ix_pos.unsqueeze(1), ix_neg.view(ix_a.size(0), -1)], dim=1)
        
        return -torch.log_softmax(d[ix_a.unsqueeze(1).expand_as(ix_all), ix_all], dim=1)[:, 0].mean()
