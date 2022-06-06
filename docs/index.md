# Welcome to pytorch-lifestream docs

## Library content

Here is a breaf overview of library with links to detail description

Library modules:
 - `ptls.data` - all you need for prepare your data for neural network feeding. [Link](data_preparation.md)
   - `ptls.data.preprocessing` - common patters for feature transformation. 
   Categorical encoding, datetime transformation, numerical feature preprocessing.
   - `ptls.data.split_tools` - convert data to ptls-data-format. Split by users and features. 
   - `ptls.data.datasets` - `torch.Dataset` interface access to the data.
 - `ptls.lightning_modules` - propose tools for training your encoders with popular frameworks like 
 CoLES, SimCLR, CPC, VICReg, ... [Link TBD](#)
 - Build your encoder with layers. [Link TBD](#):
   - `ptls.trx_encoder` - layers which makes representation for single transactions. [Link TBD](#)
   - `ptls.seq_encoder` - layers which works with sequences. [Link TBD](#)
   There are cross transactional interaction (like convolution of self attention) and reduce sequence to single vector.
   [Link TBD](#)
   - `ptls.heads` - composite layers for final embedding transformation (L2Norm, MLP, ResNet, FFT, ...). [Link TBD](#)
 - `ptls.contrastive_learning` - propose tools for contrastive learning tasks.
 There are losses, mining strategies and metrics. [Link TBD](#)

## How to guide

1. **Prepare your data**.
   - Use `Pyspark` in local or cluster mode for big dataset and `Pandas` for small.
   - Split data into required parts (train, valid, test, ...).
   - Transform features to compatible format using `Pyspark` or `Pandas` functions. 
   You can use also `ptls.data.preprocessing` for common data transformation patterns.
   - Split sequences to ptls-data format with `ptls.data.split_tools`. Save prepared data into `Parquet` format or 
   keep it in memory (`Pickle` also works).
   - Use one of available `ptls.data.datasets` which provide data access.
2. **Choose framework for encoder train**.
   - There are both supervised of unsupervised in `ptls.lightning_modules`. 
   - Keep in mind that each framework requires his own batch format.
   Tools for batch collate can be found in the selected framework package.
3. **Build encoder**.
   - All parts are available in `ptls.trx_encoder`, `ptls.seq_encoder`, `ptls.heads`.
   - You can also use pretrained layers.
4. **Train your encoder** with selected framework.
   - Some frameworks require certain options to be selected. It can be loss, metrics or something else. 
   `ptls.contrastive_learning` contains all of them. 
   - Provide data with selected framework compatible dataloader. 
   - Check the progress on tensorboard.
   - Tune hyperparameters if you need.
5. **Save trained encoder** for future use.
   - You can use it as single solution (e.g. get class label probabilities).
   - It can be a pretrained part of other neural network.
6. **Use encoder** in your project.
   - Run predict for your data and get logits, probas, scores or embeddings. 
   - Use `ptls.data` and `ptls.data.datasets` tools to keep your data transformation and collect batches for inference.

## How to create your own components

Your task may require some specific solution. 
You may create a new component for every library modules. 
There are links with description:
 - `ptls.data` - new data processing tools and datasets for new type of data. [Link TBD](#)
 - `ptls.lightning_modules` - new modules which train network for your problem. [Link TBD](#)
 - New layers for `ptls.trx_encoder`, `ptls.seq_encoder`, `ptls.heads`. [Link TBD](#)
 - `ptls.contrastive_learning` - New losses, mining strategies and metrics. [Link TBD](#)
