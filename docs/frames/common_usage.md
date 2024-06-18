# `ptls.frames` usage

`frames` means frameworks. They are collections of popular model training technics.
Each framework is a `LightningModule`. It means that you can train it with `pytorch_lightning.Trainer`.
Frameworks consume data in a special format, so a `LightningDataModule` required.
So there are three `pytorch_lightning` entities required:

- model
- data
- trainer

Trainer is a `pytorch_lightning.Trainer`. It automates training process.
You can read its description [here](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html).

We make a special `torch.nn.Dataset` implementation for each framework. All of them:

- support `map` and `iterable` version. You can use any of them. [More info about](https://pytorch.org/docs/stable/data.html#dataset-types)
- have `collate_fn` for batch collection
- consume `map` or `iterable` input as dict of feature arrays
- compatible with `ptls.frames.PtlsDataModule`

Model is usually a `seq_encoder` with an optional `head`.
We provide a model to framework assigned `LightningModule`.

## Example

This example is for CoLES framework. You can try others the same way.
See module list in `ptls.frames` submodules. Check docstring for precise parameter tuning.

### Data generation

We make a small test dataset. In real life you can use many ways to load data. See `ptls.data_load`.

```python
import torch

# Makes 1000 samples with `mcc_code` and `amount` features and seq_len randomly sampled in range (100, 200)
dataset = [{
    'mcc_code': torch.randint(1, 10, (seq_len,)),
    'amount': torch.randn(seq_len),
    'event_time': torch.arange(seq_len),  # shows order between transactions
} for seq_len in torch.randint(100, 200, (1000,))]

from sklearn.model_selection import train_test_split
# split 10% for validation
train_data, valid_data = train_test_split(dataset, test_size=0.1)
```

We can use an others sources for train and valid data.


### DataModule creation

As we choose CoLES we should use `ptls.frames.coles.ColesDataset` for `map` style
or `ptls.frames.coles.ColesIterableDataset` for `iterable`.

Our demo data is in memory, so we can use both `map` or `iterable`.
`map` style seems better because it provides better shuffle.
If data is iterable like `ptls.data_load.parquet_dataset.ParquetDataset` 
we can't use `map` style until we read it to `list`.

```python
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices

splitter=SampleSlices(split_count=5, cnt_min=10, cnt_max=20)
train_dataset = ColesDataset(data=train_data, splitter=splitter)
valid_dataset = ColesDataset(data=valid_data, splitter=splitter)
```

Created datasets returns 5 subsample with length in range (10, 20) for each user.

Now you need to create a dataloader that will collect batches. There are two ways to do this.
Manual:
```python
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate_fn,  # collate_fn from dataset
    shuffle=True,
    num_workers=4,
    batch_size=32,
)
valid_dataloader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    collate_fn=valid_dataset.collate_fn,  # collate_fn from dataset
    shuffle=False,
    num_workers=0,
    batch_size=32,
)
```

With datamodule:
```python
from ptls.frames import PtlsDataModule

datamodule = PtlsDataModule(
    train_data=train_dataset,
    train_batch_size=32,
    train_num_workers=4,
    valid_data=valid_dataset,
    valid_num_workers=0,
)
```

### Model creation

We have to create `seq_cncoder` that transforms sequences to embedding 
and create `CoLESModule` that will train `seq_encoder`.

```python
import torch.optim
from functools import partial
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule

seq_encoder = RnnSeqEncoder(
    trx_encoder=TrxEncoder(
        embeddings={'mcc_code': {'in': 10, 'out': 4}},
        numeric_values={'amount': 'identity'},
    ),
    hidden_size=16,  # this is final embedding size
)

coles_module = CoLESModule(
    seq_encoder=seq_encoder,
    optimizer_partial=partial(torch.optim.Adam, lr=0.001),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9),
)
```

### Training

Everything is ready for training. Let's create a `Trainer`.
```python
import pytorch_lightning as pl

trainer = pl.Trainer(gpus=1, max_epochs=50)
```

There are many options for `pytorch_lightning.Trainer` check docstring.

We force trainer to use one gpu with setting `gpus=1`. If you haven't gpu keep `gpus=None`.
Trainer will train our model until 50 epochs reached.

Depending on the method of creating dataloaders, the learning interface changes slightly.
With dataloaders:
```python
trainer.fit(coles_module, train_dataloader, valid_dataloader)
```

With datamodule:
```python
trainer.fit(coles_module, datamodule)
```

Result will be the same.

Now `coles_module` with `seq_encoder` are trained.

### Inference

This demo shows how to make embedding with pretrained `seq_encoder`.

`pytorch_lightning.Trainer` has a `predict` method that calls `seq_encoder.forward`.
`predict` requires `LightningModule` but `seq_encoder` is `torch.nn.Module`.
We should convert `seq_encoder` to `LightningModule`.

We can use `CoLESModule` or any other module if available. In this example we can use `coles_module` object.
Sometimes we have only `seq_encoder`, e.g. loaded from disk.
`CoLESModule` has a little overhead. There are head, loss and metrics inside.

Other way is using lightweight `ptls.frames.supervised.SequenceToTarget` module.
It can run inference with only `seq_encoder`.

```python
import torch
import pytorch_lightning as pl

from ptls.frames.supervised import SequenceToTarget
from ptls.data_load.datasets.dataloaders import inference_data_loader

inference_dataloader = inference_data_loader(dataset, num_workers=4, batch_size=256)
model = SequenceToTarget(seq_encoder)
trainer = pl.Trainer(gpus=1)
embeddings = torch.vstack(trainer.predict(model, inference_dataloader))
assert embeddings.size() == (1000, 16)
```

Final shape is depends on:

- `dataset` size, we have 1000 samples in out dataset.
- `seq_encoder.embedding_size`, we set `hidden_size=16` during `RnnSeqEncoder` creation.

## Next steps

Now you can try to change hyperparameters of `ColesDataset`, `CoLESModule` and `Trainer`.
Or try an others frameworks from `ptls.frames`.
