# Training methods

## Using Vicreg Loss with Metric Learn training

### Define the model

```python
from dltranz.seq_encoder import SequenceEncoder
from dltranz.models import Head
from dltranz.lightning_modules.emb_module import EmbModule
from dltranz.metric_learn.losses import VicregLoss

vicreg_loss = VicregLoss(sim_coeff=10.0,
                         std_coeff=10.0,
                         cov_coeff=5.0)

seq_encoder = SequenceEncoder(
    category_features=preprocessor.get_category_sizes(),
    numeric_features=["amount_rur"],
    trx_embedding_noize=0.003
)

head = Head(input_size=seq_encoder.embedding_size, hidden_layers_sizes=[256], use_batch_norm=True)


model = EmbModule(seq_encoder=seq_encoder,
                  head=head,
                  loss=vicreg_loss,
                  lr=0.01,
                  lr_scheduler_step_size=10,
                  lr_scheduler_step_gamma=0.9025)
```

### Data module

To use VicregLoss and BarlowTwinsLoss it is crucial to set `split_count=2`
```python
from dltranz.data_load.data_module.emb_data_module import EmbeddingTrainDataModule

dm = EmbeddingTrainDataModule(
    dataset=train,
    pl_module=model,
    min_seq_len=25,
    seq_split_strategy='SampleSlices',
    category_names = model.seq_encoder.category_names,
    category_max_size = model.seq_encoder.category_max_size,
    split_count=2,
    split_cnt_min=25,
    split_cnt_max=200,
    train_num_workers=16,
    train_batch_size=256,
    valid_num_workers=16,
    valid_batch_size=256
)
```

### Training

```python
import torch
import pytorch_lightning as pl

import logging

trainer = pl.Trainer(
    max_epochs=150,
    gpus=1 if torch.cuda.is_available() else 0
)

trainer.fit(model, dm)
```

