# `ptls.trames` usage

## Content
There are currently three groups of unsupervised frameworks. Groups are defined by a used technics:
- `ptls.frames.coles` - contrastive leaning on split sequences. 
Samples from original sequence are near in embedding space.
- `ptls.frames.cpc` - Contrast learning on a changing time sequence. 
Embeddings are trained to predict their future state.
- `ptls.frames.bert` - methods are inspired by nlp with transformer models

There are currently one type of supervised frameworks:
- `ptls.framed.supervised`

## `ptls.frames.coles` usage
Example with dataloader will be presented

## `ptls.frames.cpc` usage
Example with dataloader will be presented

## `ptls.frames.bert` usage
Example with dataloader will be presented

## `ptls.frames.supervised` usage
Example with dataloader will be presented

## Classes
See docstrings for classes.

- `ptls.frames.coles.CoLESModule`
- `ptls.frames.cpc.CpcModule`
- `ptls.frames.cpc.CpcV2Module`
- `ptls.frames.bert.CpcV2Module`
- `ptls.frames.bert.MLMPretrainModule`
- `ptls.frames.bert.RtdModule`
- `ptls.frames.bert.SopNspModule`
- `ptls.framed.supervised.SequenceToTarget`
