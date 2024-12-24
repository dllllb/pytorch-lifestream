from typing import Dict
import torch

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder import TrxEncoder
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder

class MultiModalSortTimeSeqEncoderContainer(torch.nn.Module):
    """Container for multimodal event sequences

    It is used when there is data on sequences of events of different modality.
    Subsequences are selected for each modality.
    The modalities are then merged together, taking into account the time order
    For each modality, its own trx_encoder is used, after which the received embedding events are fed to seq_encoder

    Parameters
        trx_encoders:
            Dict with trx encoders for each modality.
        seq_encoder_cls:
            Class of model which calculate embeddings for original raw transaction sequences.
            `seq_encoder` is trained by `CoLESModule` to get better representations of input sequences.
            ptls.nn.seq_encoder.rnn_encoder.RnnEncoder can be used.
        input_size:
            Size of transaction embeddings.
            Each trx_encoder should have the same linear_projection_size
        col_time:
            A column containing the time of events in the data to be merged.
    
    An example of use can be found at the link:
    https://github.com/dllllb/pytorch-lifestream/blob/main/ptls_tests/test_frames/test_coles/test_multimodal_coles_module.py
    """

    def __init__(
        self,
        trx_encoders: Dict[str, TrxEncoder],
        seq_encoder_cls: AbsSeqEncoder,
        input_size: int,
        is_reduce_sequence: bool = True,
        col_time: str = 'event_time',
        **seq_encoder_params
    ):
        super().__init__()
        self.trx_encoders = torch.nn.ModuleDict(trx_encoders)
        self.seq_encoder = seq_encoder_cls(
            input_size=input_size,
            is_reduce_sequence=is_reduce_sequence,
            **seq_encoder_params,
        )
        
        self.col_time = col_time
        self.input_size = input_size
    
    @property
    def is_reduce_sequence(self):
        return self.seq_encoder.is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self.seq_encoder.is_reduce_sequence = value

    @property
    def embedding_size(self):
        return self.seq_encoder.embedding_size
        
    def merge_by_time(self, x: Dict[str, torch.Tensor]):
        device = list(x.values())[1][0].device
        batch, batch_time = torch.tensor([], device=device), torch.tensor([], device=device)
        for source_batch in x.values():
            if source_batch[0] != 'None':
                batch = torch.cat((batch, source_batch[1].payload), dim=1)
                batch_time = torch.cat((batch_time, source_batch[0]), dim=1)
        
        batch_time[batch_time == 0] = float('inf')
        indices_time = torch.argsort(batch_time, dim=1)
        indices_time = indices_time.unsqueeze(-1).expand(-1, -1, self.input_size)
        batch = torch.gather(batch, 1, indices_time)
        return batch
            
    def trx_encoder_wrapper(self, x_source, trx_encoder, col_time):
        if torch.nonzero(x_source.seq_lens).size()[0] == 0:
            return x_source.seq_lens, 'None', 'None'
        return x_source.seq_lens, x_source.payload[col_time], trx_encoder(x_source)
    
    def multimodal_trx_encoder(self, x):
        res = {}
        tmp_el = list(x.values())[0]
        
        batch_size = tmp_el.payload[self.col_time].shape[0]
        length = torch.zeros(batch_size, device=tmp_el.device).int()
        
        for source, trx_encoder in self.trx_encoders.items():
            enc_res = self.trx_encoder_wrapper(x[source], trx_encoder, self.col_time)
            source_length, res[source] = enc_res[0], (enc_res[1], enc_res[2])
            length = length + source_length
        return res, length
            
    def forward(self, x, names=None, seq_len=None, **kwargs):
        if names and seq_len is not None:
            raise NotImplementedError
        x, length = self.multimodal_trx_encoder(x)
        x = self.merge_by_time(x)
        padded_x = PaddedBatch(payload=x, length=length)
        x = self.seq_encoder(padded_x, **kwargs)
        return x
