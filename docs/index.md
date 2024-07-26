# Welcome to pytorch-lifestream docs

## Library content

Here is a brief overview of library with links to the detailed descriptions.

**Library modules:**

- `ptls.preprocessing` - transforms data to `ptls`-compatible format with `pandas` or `pyspark`.
Categorical encoding, datetime transformation, numerical feature preprocessing.

- `ptls.data_load` - all that you need for prepare your data to training and validation.
    - `ptls.data_load.datasets` - PyTorch `Dataset` API implementation for data access.
    - `ptls.data_load.iterable_processing` - generator-style filters for data transformation.
    - `ptls.data_load.augmentations` - functions for data augmentation.

- `ptls.frames` - tools for training encoders with popular frameworks like 
CoLES, SimCLR, CPC, VICReg, ...
    - `ptls.frames.coles` - Contrastive leaning on sub-sequences.
    - `ptls.frames.cpc` - Contrastive learning for future event state prediction.
    - `ptls.frames.bert` - methods, inspired by NLP and transformer models.
    - `ptls.framed.supervised` - modules fo supervised training.
    - `ptls.frames.inference` - inference module.

- `ptls.nn` - layers for model creation:
    - `ptls.nn.trx_encoder` - layers to produce the representation for a single transactions.
    - `ptls.nn.seq_encoder` - layers for sequence processing, like `RNN` of `Transformer`.
    - `ptls.nn.pb` - `PaddedBatch` compatible layers, similar to `torch.nn` modules, but works with `ptls-data`.
    - `ptls.nn.head` - composite layers for final embedding transformation.
    - `ptls.nn.seq_step.py` - change the sequence along the time axis.
    - `ptls.nn.binarization`, `ptls.nn.normalization` - other groups of layers.

## How to guide

1. **Prepare your data**.
    - Use `Pyspark` in local or cluster mode for big dataset and `Pandas` for small.
    - Split data into required parts (train, valid, test, ...).
    - Use `ptls.preprocessing` for simple data preparation. 
    - Transform features to compatible format using `Pyspark` or `Pandas` functions. 
    You can also use `ptls.data_load.preprocessing` for common data transformation patterns.
    - Split sequences to `ptls-data` format with `ptls.data_load.split_tools`. Save prepared data into `Parquet` format or 
    keep it in memory (`Pickle` also works).
    - Use one of the available `ptls.data_load.datasets` to define input for the models.
2. **Choose framework for encoder train**.
    - There are both supervised of unsupervised frameworks in `ptls.frames`.
    - Keep in mind that each framework requires its own batch format.
    Tools for batch collate can be found in the selected framework package.
3. **Build encoder**.
    - All parts are available in `ptls.nn`.
    - You can also use pretrained layers.
4. **Train your encoder** with selected framework and `pytorch_lightning`.
    - Provide data with one of the DataLoaders that is compatible with selected framework. 
    - Monitor the progress on tensorboard.
    - Optionally tune hyperparameters.
5. **Save trained encoder** for future use.
    - You can use it as single solution (e.g. get class label probabilities).
    - Or it can be a pretrained part of other neural network.
6. **Use encoder** in your project.
    - Run predict for your data and get logits, probas, scores or embeddings. 
    - Use `ptls.data_load` and `ptls.data_load.datasets` tools to keep your data transformation and collect batches for inference.

## How to create your own components

It is possible create specific component for every library modules. 
Here are the links to the detailed description:

- `ptls.data_load` - new data processing tools and datasets for new types of data. [Link TBD](#)
- `ptls.frames` - new modules which train network for specific problems. [Link TBD](#)
- New layers for `ptls.nn`. [Link TBD](#)
