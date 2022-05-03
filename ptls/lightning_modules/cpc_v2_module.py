import torch

from ptls.lightning_modules.AbsModule import ABSModule
from ptls.trx_encoder import PaddedBatch
from ptls.lightning_modules.cpc_module import CPC_Loss, CpcAccuracyPL
from ptls.seq_encoder.rnn_encoder import RnnEncoder


class CpcV2Module(ABSModule):
    def __init__(self, params):
        if params['encoder_type'] != 'rnn':
            raise NotImplementedError(f'Only rnn encoder supported in CpcModule. Found {params.encoder_type}')

        super().__init__(params)
        
        # Define aggrigator
        self.agg_rnn = RnnEncoder(params.rnn.hidden_size, params.rnn_agg)
        
        # Define linear transformation for every cpc.n_forward_steps 
        self.n_forward_steps = params['cpc.n_forward_steps']
        embedding_size = params.rnn_agg.hidden_size
        linear_size = params.rnn.hidden_size
        self._linears = torch.nn.ModuleList([torch.nn.Linear(embedding_size, linear_size)
                                             for _ in range(self.n_forward_steps)])
        

    @property
    def metric_name(self):
        return 'cpc_accuracy'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def get_loss(self):
        return CPC_Loss(n_negatives=self.hparams.params['cpc.n_negatives'])

    def get_validation_metric(self):
        return CpcAccuracyPL(self._loss)

    def shared_step(self, x, y):
        """
        x: list of PaddedBatch, 
        where PaddedBatch = (
            payload = {key: [padded tensor] whit shape [batch_size, max_subseq_len]}, 
            seq_lens = (1 x batch_size)
        )
        """
        split_embeddings = []
        for padded_batch in x:
            split_embeddings.append(self._seq_encoder(padded_batch))
            
        base_embeddings = PaddedBatch(torch.stack(split_embeddings, dim=1), None)
        context_embeddings = self.agg_rnn(base_embeddings)
        
        
        me = []
        for l in self._linears:
            me.append(l(context_embeddings.payload))
        
        batch_size, max_seq_len, _ = base_embeddings.payload.shape
        
        mapped_ctx_embeddings = PaddedBatch(
            torch.stack(me, dim=3), 
            max_seq_len * torch.ones(batch_size, device=x[0].device)
        )
        return (base_embeddings, context_embeddings, mapped_ctx_embeddings), y
    
    def validation_step(self, batch, _):
        y_h, y = self.shared_step(batch, None)
        self._validation_metric(y_h, y)
        
    def training_step(self, batch, _):
        y_h, y = self.shared_step(batch, None)
        loss = self._loss(y_h, y)
        self.log('loss', loss)
        return loss
