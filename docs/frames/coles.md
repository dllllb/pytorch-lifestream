# Contrastive Learning for Event Sequences (CoLES)

Original paper: [CoLES: Contrastive Learning for Event Sequences with Self-Supervision](https://dl.acm.org/doi/10.1145/3514221.3526129)

CoLES is a framework that learn neural network to compress sequential data into a single embedding.

Imagine a credit card transaction history that can be an example of user behavioral.
Each user has his own behavioral patterns which are projected to his transaction history.
Repeatability of behavioral patterns leads to repeatability in transaction history.

CoLES exploit repeatability of patterns to make embedding. It samples a few subsequences from original sequence
and calculates an embeddings for each of them. Embeddings are assigned to a corresponding user.
Subsequences represent the same user and contain the same behavioral patterns. 
CoLES catch these patterns by making closer users embeddings. It also tries to distance different users embeddings.

Subsequences represent also original sequence, and the similarity of behavioral patterns 
allows the similarity of embeddings for original sequence and his subsequence.

**CoLES** learn:

- more 'global' representation of sequence
- embedding for each transaction is an internal state of `seq_encoder`
- embedding for all sequence is an output of `seq_encoder`

## CoLESModule
`ptls.frames.coles.CoLESModule` is a `LightningModule` with CoLES framework.
It should be parametrized by `head` and `loss`. Usually, `loss` requires a definition of `sampling_strategy`.
CoLES datamodule requires a `split_strategy`.
Combination of these parameters provides a variety of training methods.

Example:
```python
seq_encoder = ...
coles_module = CoLESModule(
    seq_encoder=seq_encoder,
    head=Head(use_norm_encoder=True),
    loss=ContrastiveLoss(
        margin=0.5,
        sampling_strategy=HardNegativePairSelector(neg_count=5),
    )
)
```

## ColesSupervisedModule

This is `ptls.frames.coles.CoLESModule` with auxiliary loss based on dataset labels.
This can be used to organize user embedding space according to known classes.
Embeddings for similar sequences still close (CoLES loss).
Embeddings with same class label are close too (auxiliary loss).

Notes:

- there can be many types of class labels, this can be targets from supervised task.
Labels for each class are provided by `ColesSupervisedDataset`.
- class labels can be missed. Auxiliary loss is calculated only for labeled data.
CoLES loss is calculated for all data.
- auxiliary loss is `l_loss` attribute of `ColesSupervisedModule` constructor.


## ColesDataset and split strategies
Use `ptls.frames.coles.ColesDataset` or `ptls.frames.coles.ColesIterableDataset` with `CoLESModule`. 
They are parametrised with `splitter` from `ptls.frames.coles.split_strategy`.
According to [CoLES paper](https://dl.acm.org/doi/10.1145/3514221.3526129) we recommend to use 
`ptls.frames.coles.split_strategy.SampleSlices` splitter.

## ColesSupervisedDataset and split strategies
Use `ptls.frames.coles.ColesDataset` or `ptls.frames.coles.ColesIterableDataset` with `ColesSupervisedModule`. 
It's parametrised by `splitter` as `ColesDataset`.

`ColesSupervisedDataset` requires a list of columns where target labels are stored (`cols_classes` attribute).
It is used to provide these labels to dataloader.

## Coles losses and sampling strategies
Use classes from:

- `ptls.frames.coles.losses`
- `ptls.frames.coles.sampling_strategies`

Usage recommendations:

- Auxiliary class labels don't change because they are client related. This means that you can use losses with memory
to learn class centers in embedding space for `l_loss` in `ColesSupervisedModule`.
Losses without memory calculate class center for batch.
- Don't use losses with memory as CoLES loss, cause Coles labels valid only in batch.
CoLES labels are aranged over batch, so e.g. 0-label correspond different clients in different batches.


## Head selection
Use `ptls.nn.Head`.
Default head has only l2-norm layer (`Head(use_norm_encoder=True)`).

Head with MLP inside realise `projection head` concept from SimCLR.

## Classes
See docstrings for classes.

- `ptls.frames.coles.ColesDataset`
- `ptls.frames.coles.ColesIterableDataset`
- `ptls.frames.coles.CoLESModule`

- `ptls.frames.coles.split_strategy`
- `ptls.frames.coles.losses`
- `ptls.frames.coles.sampling_strategies`
