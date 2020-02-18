"""
This code taken from    https://github.com/locuslab/trellisnet
"""
import torch

from dltranz.trellisnet.trellisnet import TrellisNet
from dltranz.trx_encoder import PaddedBatch


class TrellisNetEncoder(torch.nn.Module):
    def __init__(self, enc_input_size, params):
        super().__init__()

        self.nhid = params['nhid']
        self.nout = params['nout']
        self.model = TrellisNet(**params)

    def forward(self, x: PaddedBatch):
        batch_size = x.payload.size()[0]
        hidden = self.init_hidden(batch_size, x.payload.device)

        t = x.payload.transpose(1, 2)
        out = self.model(t, hidden, aux=False)

        return PaddedBatch(out[0], x.seq_lens)

    def init_hidden(self, bsz, device):
        h_size = self.nhid + self.nout
        weight = next(self.parameters()).data
        return (weight.new(bsz, h_size, 1).zero_().to(device),
                weight.new(bsz, h_size, 1).zero_().to(device))
