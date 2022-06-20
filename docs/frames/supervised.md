# Supervised lightning modules

## `ptls.frames.supervised.SequenceToTarget` in supervised mode

`SequenceToTarget` is a lightning module for supervised training.
This module assumes a target for sequence.

There can be a some types of supervised task, like classification of regression.
`SequenceToTarget` parameters allows to fit this module to your task.

`SequenceToTarget` requires `seq_encoder`, `head`, `loss` and `metrics`.

Se en examples of usage in `SequenceToTarget` docstring.

Layers from `seq_encoder`, `head` can be randomly initialized or pretrained.


## `ptls.frames.supervised.SequenceToTarget` in inference mode

You may just provide pretrained `seq_encoder` to `SequenceToTarget` and 
use `trainer.predict` to get embeddings from pretrained `seq_encoder`.

## Classes

See docstrings for classes.

- `ptls.frames.supervised.SequenceToTarget`
