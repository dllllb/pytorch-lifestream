import torch
from ptls.data_load.padded_batch import PaddedBatch

class MultiModalSortTimeSeqEncoderContainer(torch.nn.Module):
    def __init__(self,
                 trx_encoders,
                 seq_encoder_cls, 
                 input_size,
                 is_reduce_sequence=True,
                 col_time='event_time',
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
    
    @property
    def is_reduce_sequence(self):
        return self.seq_encoder.is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self.seq_encoder.is_reduce_sequence = value

    @property
    def embedding_size(self):
        return self.seq_encoder.embedding_size
    
    def get_tensor_by_indices(self, tensor, indices):
        batch_size = tensor.shape[0]
        return tensor[:, indices, :][torch.arange(batch_size), torch.arange(batch_size), :, :]
        
    def merge_by_time(self, x):
        device = list(x.values())[1][0].device
        batch, batch_time = torch.tensor([], device=device), torch.tensor([], device=device)
        for source_batch in x.values():
            if source_batch[0] != 'None':
                batch = torch.cat((batch, source_batch[1].payload), dim=1)
                batch_time = torch.cat((batch_time, source_batch[0]), dim=1)
        
        batch_time[batch_time == 0] = float('inf')
        indices_time = torch.argsort(batch_time, dim=1)
        batch = self.get_tensor_by_indices(batch, indices_time)
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
            
    def forward(self, x):
        x, length = self.multimodal_trx_encoder(x)
        x = self.merge_by_time(x)
        padded_x = PaddedBatch(payload=x, length=length)
        x = self.seq_encoder(padded_x)
        return x