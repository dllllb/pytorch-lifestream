import torch
import torch.nn as nn

from ptls.data_load.padded_batch import PaddedBatch

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
        self.is_inference = False
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
    
    def merge_by_time_by_index(self, x, indices):
        m, n, _ = indices.size()
        sources = list(x.keys())
        emb_size = x[sources[0]][1].payload.shape[2]
        output_embeddings = torch.zeros(m, n, emb_size, device=indices.device)

        mod_indices = indices[:, :, 1].long()
        emb_indices = indices[:, :, 0].long()

        for mod in range(len(sources)):
            mod_mask = mod_indices == mod
            embeddings = x[sources[mod]][1].payload

            gather_indices =mod_mask.unsqueeze(-1).expand(-1, -1, embeddings.size(2)) * emb_indices.unsqueeze(-1)
            selected_embeddings = torch.gather(embeddings, 1, gather_indices)
            output_embeddings = torch.where(mod_mask.unsqueeze(-1), selected_embeddings, output_embeddings)

        return output_embeddings
            
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
        x, time_index = x
        x, length = self.multimodal_trx_encoder(x)
        if self.is_inference:
            x = self.merge_by_time(x)
        else:
            x = self.merge_by_time_by_index(x, time_index)
        padded_x = PaddedBatch(payload=x, length=length)
        x = self.seq_encoder(padded_x)
        return x
