# Feature naming and types

## Feature types

Information about transaction features are stored as array in dictionary.

There are feature types:

- Sequential feature - is a `np.ndarray` or `torch.tensor` of shape `(seq_len,)`
    - for categorical features contains category indexes with type `long`
    - for numerical features contains feature value with type `float`
- Scalar values. It can be `target`, `id`, `labels` or `scalar features`.
Types are depends on purpose. Type should be compatible with torch if value will be fed into neural network
- Array values. It also can be `target`, `id`, `labels` or `vector features`.
Type is `np.ndarray` or `torch.tensor`.

Sequential features correspond user's transactions.
The length of each user's sequential feature is equal to the length of the entire sequence.
The order of each user's sequential feature is the same as sequence order.
Sequential feature length `seq_len` may vary from user to user.

Array features have a constant shape. This shape is the same for all users.

This why we use `pad_sequence` which align length for sequential features and `stack` for array features
during batch collection.

`ptls` extract only sequential features for unsupervised task and additional target for the supervised task.
Other fields used during preprocessing and inference.

## Feature names

The main purpose of the feature naming convention is sequential and array features distinguish.
They both are `np.ndarray` or `torch.tensor` and we can't use data type for distinguish.

It's important to know feature type because:

- sequential align lengths with `pad_sequence`, arrays use `stack` during batch collection.
- only sequential features used to get length of entire sequence
- only sequential features are augmented by timeline modifications like slice, trx dropout or shuffle

We introduce naming rules to solve type discrimination problems.
All arrays which are not sequential should have `target` prefix in feature name.
Otherwise, they can be processed as sequential and may be corrupted.

```python
# correct example
x = {
    'mcc': torch.tensor([1, 2, 3, 4]),
    'amount': torch.tensor([0.1, 2.0, 0.3, 4.0]),
    'target_bin': 1,
    'target_distribution': torch.tensor([0.1, 0.0, 0.9]),
}

# wrong example
x = {
    'mcc': torch.tensor([1, 2, 3, 4]),
    'amount': torch.tensor([0.1, 2.0, 0.3, 4.0]),
    'bin': 1,
    'distribution': torch.tensor([0.1, 0.0, 0.9]),
}
```

`target` prefix are mandatory only for array features.

Sometimes we need a time sequence. It used fo trx correct order, for time features and for some splits.
We expect that transaction timestamp stored in `event_time` field.

## Naming rules

- all arrays which are not sequential should have `target` prefix in feature name.
- `event_time` fields contains transaction timestamps sequence.

## Feature rename

You can use `ptls.data_load.iterable_processing.FeatureRename` during data read pipeline
to fit your feature names with ptls naming convention.

```python
x = [{
    'mcc': torch.tensor([1, 2, 3, 4]),
    'amount': torch.tensor([0.1, 2.0, 0.3, 4.0]),
    'bin': 1,
    'distribution': torch.tensor([0.1, 0.0, 0.9]),
} for _ in range(10)]

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import FeatureRename
dataset = MemoryMapDataset(
    data=x,
    i_filters=[FeatureRename({'distribution': 'target_distribution', 'bin': 'target_bin'})]
)

print(dataset[0])
```

## Code usage

Need to take into account the type of features and the use of naming rules is in the classes:

- `ptls.data_load.feature_dict.FeatureDict`
- `ptls.data_load.padded_batch.PaddedBatch`
- `ptls.data_load.utils.collate_feature_dict`

All methods are tested with all types of features.

| Type           | FeatureDict               | PaddedBatch   | collate_feature_dict | is_seq |
| -------------- | ------------------------- | ------------- | -------------------- | ------ | 
| scalar int     | `int`                     | 1-d `tensor`  | `torch.IntTensor`    |   X    |
| target int     | `int`                     | 1-d `tensor`  | `torch.IntTensor`    |   X    |
| scalar float   | `float`                   | 1-d `tensor`  | `torch.FloatTensor`  |   X    |
| scalar str     | `str`                     | 1-d `ndarray` | `np.array`           |   X    |
| list           | `list`                    | 1-d `ndarray` | `np.array`           |   X    |
| sequential     | 1-d `ndarray` or `tensor` | 2-d `tensor`  | `pad_sequence`       |   V    |
| sequential et  | 1-d `ndarray` or `tensor` | 2-d `tensor`  | `pad_sequence`       |   V    |
| target array   | 1-d `ndarray` or `tensor` | 2-d `tensor`  | `stack`              |   X    |
