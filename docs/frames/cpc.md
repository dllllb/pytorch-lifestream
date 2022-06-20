# Contrastive Predictive Coding (CPC)

Original paper: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)

CPC is a framework that learn neural network to predict the future state of sequence.

CPC splits original sequence into small parts. Smallest part is a one transaction.
`trx_encoder` or `seq_encoder` make a representation for each small part.
So the original transaction sequence turns into sequence of embeddings.

CPC tries to predict a next embedding in sequence. It takes into account some history of embeddings.
Loss is contrastive, it uses random negative samples to avoid a trivial solution.

**CPC** learn:

- more 'local' representation of sequence
- embedding for each transaction is a `z` state for `CpcModule`
- embedding for small parts of sequence is a `z` state for `CpcV2Module`
- embedding for all sequence is `c` - context state of CPC encoder


## Usage
`ptls.frames.cpc.CpcModule` and `ptls.frames.cpc.CpcV2Module` is a `LightningModule` with CPC framework.
It should be parametrized by `n_negatives` and `n_forward_steps` parameters.
`CpcV2Module` parametrized also by `aggregator` network.
CPC V2 datamodule requires a split strategy.

Example:
```python
split_strategy = ...
datamodule = some_function(data, split_strategy)
seq_encoder = ...
coles_module = CpcModule(
    seq_encoder=seq_encoder,
    loss=CPC_Loss(
        n_negatives=16,
        n_forward_steps=3,
    )
)
trainer = pl.Trainer()
trainer.fit(coles_module, datamodule)
```

## CpcDataModule and split strategies

- `ptls.data_load.data_module.cpc_data_module.CpcDataModuleTrain`


## Classes
See docstrings for classes.

- `ptls.frames.cpc.CpcModule`
- `ptls.frames.cpc.CpcV2Module`
