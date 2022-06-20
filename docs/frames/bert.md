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

## Lightning modules:

- `ptls.frames.bert.MLMPretrainModule`
- `ptls.frames.bert.RtdModule`
- `ptls.frames.bert.SopNspModule`

## Data modules

- `ptls.data_load.data_module.mlm_data.MLMDataset`
- `ptls.data_load.data_module.rtd_data_module.RtdDataModuleTrain`
- `ptls.data_load.data_module.sop_data_module.SopDataModuleTrain`
- `ptls.data_load.data_module.nsp_data_module.NspDataModuleTrain`
