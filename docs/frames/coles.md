# Contrastive Learning for Event Sequences (CoLES)

Original paper: [CoLES: Contrastive Learning for Event Sequences with Self-Supervision](https://dl.acm.org/doi/10.1145/3514221.3526129)

CoLES is a framework that learn neural network to compress sequential data into a single embedding.

Imagine a credit card transaction history that can be an example of user behavioral.
Each user have his own behavioral patterns which are projected to his transaction history.
Repeatability of behavioral patterns lead to repeatability in transaction history.

CoLES exploit repeatability of patterns to make embedding. It samples a few subsequences from original sequence
and calculates an embeddings for each of them. Embeddings are assigned to his user.
Subsequences represent the same user and contain the same behavioral patterns. 
CoLES catch these patterns by making closer users embeddings. It also tries to distance different users embeddings.

Subsequences represent also original sequence, and the similarity of behavioral patterns 
allows the similarity of embeddings for original sequence and his subsequence.

**CoLES** learn:

- more 'global' representation of sequence
- embedding for each transaction is an internal state of `seq_encoder`
- embedding for all sequence is an output of `seq_encoder`

## Usage
`ptls.frames.coles.CoLESModule` is a `LightningModule` with CoLES framework.
It should be parametrized by `head` and `loss`. Usually, `loss` requires a definition of `sampling_strategy`.
CoLES datamodule requires a `split_strategy`.
Combination of these parameters provides a variety of training methods.

Example:
```python
split_strategy = ...
datamodule = some_function(data, split_strategy)
seq_encoder = ...
coles_module = CoLESModule(
    seq_encoder=seq_encoder,
    head=Head(use_norm_encoder=True),
    loss=ContrastiveLoss(
        margin=0.5,
        sampling_strategy=HardNegativePairSelector(neg_count=5),
    )
)
trainer = pl.Trainer()
trainer.fit(coles_module, datamodule)
```

## ColesDataModule and split strategies

- `ptls.data_load.data_module.coles_data_module.ColesDataModuleTrain`
- `ptls.frames.coles.split_strategy`

## Coles losses and sampling strategies

- `ptls.frames.coles.losses`
- `ptls.frames.coles.sampling_strategies`

## Head selection

- `ptls.nn.Head`

## Classes
See docstrings for classes.

- `ptls.frames.coles.CoLESModule`
