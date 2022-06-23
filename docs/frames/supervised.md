# Supervised learning
`ptls.frames.supervised.SeqToTargetDataset` and `ptls.frames.supervised.SequenceToTarget`
for supervised learning

## `SeqToTargetDataset`

Works similar as other datasets described in [common patterns](common_usage.md)

Source data should have a scalar field with target value.

Example:
```python
    dataset = SeqToTargetDataset([{
        'mcc_code': torch.randint(1, 10, (seq_len,)),
        'amount': torch.randn(seq_len),
        'event_time': torch.arange(seq_len),  # shows order between transactions
        'target': target,
    } for seq_len, target in zip(
        torch.randint(100, 200, (4,)),
        [0, 0, 1, 1],
    )], target_col_name='target')
    dl = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=dataset.collate_fn)
    x, y = next(iter(dl))
    torch.testing.assert_close(y, torch.LongTensor([0, 0, 1, 1]))
```

## `SequenceToTarget` in supervised mode

`SequenceToTarget` is a lightning module for supervised training.
This module assumes a target for sequence.

There can be a some types of supervised task, like classification of regression.
`SequenceToTarget` parameters allows to fit this module to your task.

`SequenceToTarget` requires `seq_encoder`, `head`, `loss` and `metrics`.

Se en examples of usage in `SequenceToTarget` docstring.

Layers from `seq_encoder`, `head` can be randomly initialized or pretrained.


## `SequenceToTarget` in inference mode

You may just provide pretrained `seq_encoder` to `SequenceToTarget` and 
use `trainer.predict` to get embeddings from pretrained `seq_encoder`.

## Classes

See docstrings for classes.

- `ptls.frames.supervised.SeqToTargetDataset`
- `ptls.frames.supervised.SequenceToTarget`
