# Bidirectional Encoder Representations from Transformers (BERT)

Transformer based models pretrained with unsupervised task are state-of-the-art in NLP.
We implement them for sequential data.

Pretrain tasks are implemented:

- Replaced Token Detection (RTD) from [ELECTRA](https://arxiv.org/abs/2003.10555)
- Next Sequence Prediction (NSP) from [BERT](https://arxiv.org/abs/1810.04805)
- Sequences Order Prediction (SOP) from [ALBERT](https://arxiv.org/abs/1909.11942)
- Masked Language Model (MLM) from [ROBERTA](https://arxiv.org/abs/1907.11692)

All of these tasks learn internal structure of the data and use it to make representation.

**NSP**, **RTD** learn:

- 'global' representation of sequence
- embedding for each transaction is an internal state of `seq_encoder`
- embedding for all sequence is an output of `seq_encoder`

**RTD** learn:

- 'local' representation of sequence
- embedding for each transaction is an internal state of `seq_encoder`
- embedding for all sequence available but aren't learned

**MLM** learn:

- 'local' representation of sequence
- embedding for each transaction from `trx_encoder`
- pretrained MLM transformer as `seq_encoder`, CLS token aren't learned

## MLM
`ptls.frames.bert.MLMPretrainModule` is a lightning module.

`ptls.frames.bert.MlmDataset`, `ptls.frames.bert.MlmIterableDataset` is a compatible datasets.
`ptls.frames.bert.MlmIndexedDataset` is also compatible with MLM.
`MlmDataset` dataset sample one slice for one user. `MlmIndexedDataset` sample all possible slices for each user.
`MlmIndexedDataset` index the data this because it hasn't iterable-style variant.

## RTD
`ptls.frames.bert.RtdModule` is a lightning module.

`ptls.frames.bert.RtdDataset`, `ptls.frames.bert.RtdIterableDataset` is a compatible datasets.


## SOP
`ptls.frames.bert.SopModule` is a lightning module.

`ptls.frames.bert.SopDataset`, `ptls.frames.bert.SopIterableDataset` is a compatible datasets.
Requires `splitter` from `ptls.frames.coles.split_strategy`

## NSP
`ptls.frames.bert.NspModule` is a lightning module.

`ptls.frames.bert.NspDataset`, `ptls.frames.bert.NspIterableDataset` is a compatible datasets.
Requires `splitter` from `ptls.frames.coles.split_strategy`

## Classes
See docstrings for classes.

- `ptls.frames.bert.MlmDataset`
- `ptls.frames.bert.MlmIterableDataset`
- `ptls.frames.bert.MlmIndexedDataset`
- `ptls.frames.bert.RtdDataset`
- `ptls.frames.bert.RtdIterableDataset`
- `ptls.frames.bert.SopDataset`
- `ptls.frames.bert.SopIterableDataset`
- `ptls.frames.bert.NspDataset`
- `ptls.frames.bert.NspIterableDataset`


- `ptls.frames.bert.MLMPretrainModule`
- `ptls.frames.bert.RtdModule`
- `ptls.frames.bert.SopModule`
- `ptls.frames.bert.NspModule`
