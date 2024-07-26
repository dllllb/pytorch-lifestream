# How to use `ptls.data_load.datasets` datasets

Here are the datasets (`torch.utils.data.Dataset`) which assure interface to the data.

For data prepared in memory use:

1. `MemoryMapDataset` with `i_filters`
2. `AugmentationDataset` with `f_augmentations` if needed
3. endpoint map dataset from `ptls.frames`

For small (map mode) parquet data use:

1. `ParquetDataset` with `i_filters`
2. `PersistDataset`
3. `AugmentationDataset` with `f_augmentations` if needed
4. endpoint map dataset from `ptls.frames`

For large (iterable mode) parquet data use:

1. `ParquetDataset` with `i_filters`
2. `AugmentationIterableDataset` with `f_augmentations` if needed
3. endpoint iterable dataset from `ptls.frames`

`DuckDbDataset` with `i_filters` can be used for large (iterable mode) volume data. It does not require preprocessing step for record grouping. Any input format supported by DuckDB can be used.

Other dataset order and combination are possible but not tested.

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

## map and iterable

There are the types of torch datasets.
More info about [dataset types](https://pytorch.org/docs/stable/data.html#dataset-types) to understand `map` and `iterable`.

Dataloader choose a way of iteration based on type his dataset.
In out pipeline Dataloader works with endpoint dataset from `ptls.frames`.
So the type of endpoint dataset from `ptls.frames` choose a way of iteration.

Map dataset provide better shuffle. Iterable dataset requires less memory.

> **Warning** for multiprocessing dataloader
> 
> Each worker use the same source data.
> 
> Map dataloader knows dataset `len` and uses `sampler` to randomly split all indexes from `range(o, len)` between workers.
> So each worker use his own part of data.
> 
> Iterable dataloader can just iterate over the source data. In default case each worker iterate the same data
> and **output are multiplied** by worker count.
>
> To avoid this iterable datasets should implement a way to split a data between workers.

Multiprocessing split implementation:

- `ParquetDataset` implements split logic, hence, it works correctly in worker processes
- `i_filters` and `f_augmentations` don't contain data, hence, it works correctly in worker processes
- Iterable endpoint datasets works correctly with iterable source
- Iterable endpoint datasets **multiply data with map source**
- `PersistDataset` iterate input during initialisation. Usually this happens out of dataloader in single main process, hence, it works correctly in worker processes.

## `i_filters` and `f_augmentations`

- `i_filters` - iterable filters
- `f_augmentations` - augmentation functions

### Filters

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
Many datasets in `ptls.data_load.datasets` support `i_filters`. 
They takes `i_filters` as list of `iterable_processing` objects.

### Augmentations

Sometimes we have to change items from train data. This is what `augmentations` do.
They are in `ptls.data_load.augmentations`.

Example:
```python
from ptls.data_load.augmentations import RandomSlice

f_augmentation = RandomSlice(min_len=4, max_len=10)
for rec in data_list:
    new_rec = f_augmentation(rec)
    print(new_rec)
```

Here `RandomSlice` augmentation take a random slice from source record.

### Compare

| `i_filter` | `f_augmentation` |
| ---------- | ---------------- |
| May change record. Result is always the same | May change record. Result is random |
| Place it be before persist stage to run it once and save total cpu resource | Don't place it before persist stage because it kills the random |
| Can delete items | Can not delete items |
| Can yield new items | Can not create new items |
| Works as a generator and requires iterable processing | Works as a function can be both map or iterable |

## In memory data

In memory data is common case. Data can be a list or generator with feature dicts.

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

`ptls.data_load.datasets.MemoryMapDataset`:

- implements `map` dataset
- iterates over the data and stores it in an internal list
- looks like a list

`ptls.data_load.datasets.MemoryIterableDataset`: 

- implements `iterable` dataset
- just iterates over the data
- looks like a generator

> **Warning**
> 
> Currently `MemoryIterableDataset` don`t support initial data split between workers.
> We don't recommend use it without modification.

Both datasets support any kind of input: list or generator.
As all datasets supports tha same format (list or generator) as input and output they can be chained.
This makes sense for some cases.

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

For large amount of data `pyspark` is a possible engine to prepare data and convert it in feature dict format.
See `demo/pyspark-parquet.ipynb` with example of data preprocessing with `pyspark` and parquet file preparation.

`ptls.data_load.datasets.ParquetDataset` is a dataset which reads parquet files with feature dicts.

`ptls.data_load.datasets.ParquetDataset`: 

- implements `iterable` dataset
- works correct with multiprocessing dataloader
- looks like a generator
- supports `i_filters`

You can feed `ParquetDataset` directly to dataloader for `iterable` way of usage.
You can combine `ParquetDataset` with `MemoryMapDataset` to `map` way of usage.

`ParquetDataset` requires parquet file names. Usually `spark` saves many parquet files for one dataset, 
depending on the number of partitions.
You can get all file names with `ptls.data_load.datasets.ParquetFiles` or `ptls.data_load.datasets.parquet_file_scan`.
Many files for one dataset allows you to:

- control amount of data by reading more or less files
- split data on train, valid, test

## Persist dataset

`ptls.data_load.datasets.PersistDataset` store items from source dataset to the memory.

If your source data is iterator (like python generator or `ParquetDataset`) 
all `i_filters` will be called each time when you access the data.
Persist the data into memory and `i_filters` will be called once.
Much memory may be used to store all dataset items.
Data access is faster.

Persisted iterator have `len` and can be randomly accessed by index.

## Augmentations

Class `ptls.data_load.datasets.AugmentationDataset` is a way to apply augmentations.
Example:
```python
from ptls.data_load.datasets import AugmentationDataset, PersistDataset, ParquetDataset
from ptls.data_load.augmentations import AllTimeShuffle, DropoutTrx

train_data = AugmentationDataset(
    f_augmentations=[
        AllTimeShuffle(),
        DropoutTrx(trx_dropout=0.01),
    ],
    data=PersistDataset(
        data=ParquetDataset(...),
    ),
)
```

Here we are using iterable `ParquetDataset` as the source, loading it into memory using `PersistDataset`. 
Then, each time we access the data, we apply two augmentation functions to the items stored in the `PersistDataset`.

`AugmentationIterableDataset` works in iterable mode. In this case example will be like this:
```python
train_data = AugmentationIterableDataset(
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
- `ptls.data_load.datasets.PersistDataset`

See docstrings for functions:

- `ptls.data_load.datasets.parquet_file_scan`
