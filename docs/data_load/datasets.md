# `ptls.data_load.datasets`

Here are the datasets (`torch.utils.data.Dataset`) which assure interface to the data.

## Simple example

Dict features in a list is a simple example of data.
Python's `list` have the same interface as `torch.Dataset`, so you can just provide it to dataloader.

```python
import torch
from ptls.data_load.utils import collate_feature_dict

data_list = [
    {
        'mcc': torch.arange(seq_len),
        'id': f'user_{i}'
    }
    for i, seq_len in enumerate([4, 3, 6])
]

dl = torch.utils.data.DataLoader(
    dataset=data_list,
    collate_fn=collate_feature_dict,
    batch_size=2,
)

for batch in dl:
    print(batch.payload, batch.seq_lens, sep='\n')
```

In this example we use simple list as dataset.

Sometimes you need to make changes in the dataset. We propose a filter approach for this.

## `i_filters` - iterable filters

`ptls` propose filters for dataset transformation. All of them are in `ptls.data_load.iterable_processing`.
These filter implemented in generator-style. Call filter object to get generator with modified records.

```python
from ptls.data_load.iterable_processing import SeqLenFilter

i_filter = SeqLenFilter(min_seq_len=4)
for rec in i_filter(data_list):
    print(rec)
```

There were 3 examples in the list, it became 2 cause SeqLenFilter drop short sequence.

Many kinds of filters possible: dropping records, multiply records, records transformation.

`i_filters` can be chained. Datasets provide a convenient way to do it. 
All datasets in `ptls.data_load.datasets` support `i_filters`. 
They takes `i_filters` as list of `iterable_processing` objects.

## In memory data

In memory data is common case. Data can a list or generator with feature dicts.

```python
import torch
import random

data_list = [
    {
        'mcc': torch.arange(seq_len),
        'id': f'user_{i}'
    }
    for i, seq_len in enumerate([4, 3, 6])
]

def data_gen(n):
    for i in range(n):
        seq_len = random.randint(4, 8)
        yield {
            'mcc': torch.arange(seq_len),
            'id': f'user_{i}'
        }
```

More info about [dataset types](https://pytorch.org/docs/stable/data.html#dataset-types) to understand `map` and `iterable`.

`ptls.data_load.datasets.MemoryMapDataset`:

- implements `map` dataset
- iterates over the data and stores it in an internal list
- looks like a list

`ptls.data_load.datasets.MemoryIterableDataset`: 

- implements `iterable` dataset
- just iterates over the data
- looks like a generator

Both datasets support any kind of input: list or generator.
As all datasets supports tha same format (list or generator) as input and output they can be chained.
This make sense for some cases.

Data pipelines:

- `list` input with `MemoryMapDataset` - dataset keep modified with `i_filters` data. Original data is unchanged.
`i_filters` applied once for each record. This assures fast item access but slow start.
You should wait until all data are passed through `i_filters`.
- `generator` input with `MemoryMapDataset` - dataset iterate over generator and keep the result in memory.
More memory are used, but faster access is possible. `i_filters` applied once for each record.
Freezes items taken from generator if it uses some random during generation.
- `list` with `MemoryIterableDataset` - take more times for data access cause `i_filters` 
applied during each record access (for each epoch). Faster start,
you don't wait until all data are passed through `i_filters`.
- `generator` input with `MemoryIterableDataset` - generator output modified with `i_filters` data.
Less memory used. Infinite dataset is possible.

Example:
```python
import torch
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.iterable_processing import SeqLenFilter, FeatureRename


data_list = [
    {
        'mcc': torch.arange(seq_len),
        'id': f'user_{i}'
    }
    for i, seq_len in enumerate([4, 3, 6, 2, 8, 3, 5, 4])
]

dataset = MemoryMapDataset(
    data=data_list,
    i_filters=[
        SeqLenFilter(min_seq_len=4),
        FeatureRename({'id': 'user_id'}),
    ]
)

dl = torch.utils.data.DataLoader(
    dataset=dataset,
    collate_fn=collate_feature_dict,
    batch_size=10,
)

for batch in dl:
    print(batch.payload, batch.seq_lens, sep='\n')

```

## Parquet file read

For large amount of data `pyspark` is possible engine to prepare data and convert it in feature dict format.
See `demo/pyspark-parquet.ipynb` with example of data preprocessing with `pyspark` and parquet file preparation.

`ptls.data_load.datasets.ParquetDataset` is a dataset which reads parquet files with feature dicts.

`ptls.data_load.datasets.ParquetDataset`: 

- implements `iterable` dataset
- looks like a generator
- supports `i_filters`

You can feed `ParquetDataset` directly fo dataloader for `iterable` way of usage.
Cou can combine `ParquetDataset` with `MemoryMapDataset` to `map` way of usage.

`ParquetDataset` requires parquet file names. Usually `spark` saves many parquet files for one dataset, 
depending on the number of partitions.
You can get all file names with `ptls.data_load.datasets.ParquetFiles` or `ptls.data_load.datasets.parquet_file_scan`.
Many files for one dataset allows you to:

- control amount of data by reading more or less files
- split data on train, valid, test

## Augmentations

Sometimes we have to change an items from train data. This is `augmentations`.
It's similar to `iterable_processing` they also change a record.
But `iterable_processing` returns the same result. 
`augmentations` result changes every time you call it cause if internal random.

Example: `ptls.data_load.iterable_processing.ISeqLenLimit` keep last N transactions. 
`ptls.data_load.augmentations.RandomSlice` take N transactions with random start position.
Both return N transactions. `ISeqLenLimit` returns the same transactions every time.
`RandomSlice` returns new transactions every time.

If you use `map` dataset `augmentations` should be after iter-to-map stage.

Class `ptls.data_load.datasets.AugmentationDataset` is a way to apply augmentations.
Example:
```python
from ptls.data_load.datasets import AugmentationDataset, MemoryMapDataset, ParquetDataset
from ptls.data_load.augmentations import AllTimeShuffle, DropoutTrx

train_data = AugmentationDataset(
    f_augmentations=[
        AllTimeShuffle(),
        DropoutTrx(trx_dropout=0.01),
    ],
    data=MemoryMapDataset(
        data=ParquetDataset(...),
    ),
)
```

Here we are using iterable `ParquetDataset` as the source, loading it into memory using `MemoryMapDataset`. 
Then, each time we access the data, we apply two augmentation functions to the items stored in the `MemoryMapDataset`.

`AugmentationDataset` also works in iterable mode. Previous example will be like this:
```python
train_data = AugmentationDataset(
    f_augmentations=[
        AllTimeShuffle(),
        DropoutTrx(trx_dropout=0.01),
    ],
    data=ParquetDataset(...),
)
```

## Classes and functions
See docstrings for classes:

- `ptls.data_load.datasets.MemoryMapDataset`
- `ptls.data_load.datasets.MemoryIterableDataset`
- `ptls.data_load.datasets.ParquetFiles`
- `ptls.data_load.datasets.ParquetDataset`

See docstrings for functions:

- `ptls.data_load.datasets.parquet_file_scan`
