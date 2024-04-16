import torch
import torch.distributed as dist

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grad_outs):
        # if os.environ.get('REDUCE_GRADS'):
        # grad_outs = torch.stack(grad_outs)
        # dist.all_reduce(grad_outs)
        return grad_outs[dist.get_rank()]

def all_gather_and_cat(tensor):
    return torch.cat(AllGather.apply(tensor))

