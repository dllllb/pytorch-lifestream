![ptls-logo](ptls-banner.png)


[![GitHub license](https://img.shields.io/github/license/dllllb/pytorch-lifestream.svg)](https://github.com/dllllb/pytorch-lifestream/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/pytorch-lifestream.svg)](https://badge.fury.io/py/pytorch-lifestream)
[![GitHub issues](https://img.shields.io/github/issues/dllllb/pytorch-lifestream.svg)](https://github.com/dllllb/pytorch-lifestream/issues)
[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/pytorch_lifestream)

`pytorch-lifestream` or ptls a library built upon [PyTorch](https://pytorch.org/) for building embeddings on discrete event sequences using self-supervision. It can process terabyte-size volumes of raw events like game history events, clickstream data, purchase history or card transactions.

It supports various methods of self-supervised training, adapted for event sequences:

- Contrastive Learning for Event Sequences ([CoLES](https://arxiv.org/abs/2002.08232))
- Contrastive Predictive Coding ([CPC](https://arxiv.org/abs/1807.03748))
- Replaced Token Detection (RTD) from [ELECTRA](https://arxiv.org/abs/2003.10555)
- Next Sequence Prediction (NSP) from [BERT](https://arxiv.org/abs/1810.04805)
- Sequences Order Prediction (SOP) from [ALBERT](https://arxiv.org/abs/1909.11942)
- Masked Language Model (MLM) from [ROBERTA](https://arxiv.org/abs/1907.11692)

It supports several types of encoders, including Transformer and RNN. It also supports many types of self-supervised losses.

The following variants of the contrastive losses are supported:

- Contrastive loss ([paper](https://doi.org/10.1109/CVPR.2006.100))
- Triplet loss ([paper](https://arxiv.org/abs/1412.6622))
- Binomial deviance loss ([paper](https://arxiv.org/abs/1407.4979))
- Histogramm loss ([paper](https://arxiv.org/abs/1611.00822))
- Margin loss ([paper](https://arxiv.org/abs/1706.07567))
- VICReg loss ([paper](https://arxiv.org/abs/2105.04906))

## Install from PyPi

```sh
pip install pytorch-lifestream
```

## Install from source

```sh
# Ubuntu 20.04

sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv sync --dev # install packages exactly as specified in Pipfile.lock
pipenv shell
pytest

```

## Demo notebooks

We have a demo notebooks here, some of them:

- Supervised model training [notebook](tutorials/notebooks/supervised-sequence-to-target.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dllllb/pytorch-lifestream/blob/master/tutorials/notebooks/supervised-sequence-to-target.ipynb)
- Self-supervided training and embeddings for downstream task [notebook](tutorials/notebooks/coles-emb.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dllllb/pytorch-lifestream/blob/master/tutorials/notebooks/coles-emb.ipynb)
- Self-supervided embeddings in CatBoost [notebook](tutorials/notebooks/coles-catboost.ipynb)
- Self-supervided training and fine-tuning [notebook](tutorials/notebooks/coles-finetune.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dllllb/pytorch-lifestream/blob/master/tutorials/notebooks/coles-finetune.ipynb)
- Self-supervised TrxEncoder only training with Masked Language Model task and fine-tuning [notebook](tutorials/notebooks/mlm-emb.ipynb)
- Pandas data preprocessing options [notebook](tutorials/notebooks/preprocessing-demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dllllb/pytorch-lifestream/blob/master/tutorials/notebooks/preprocessing-demo.ipynb)
- PySpark and Parquet for data preprocessing [notebook](tutorials/notebooks/pyspark-parquet.ipynb)
- Fast inference on large dataset [notebook](tutorials/notebooks/extended_inference.ipynb)
- Supervised multilabel classification [notebook](tutorials/notebooks/multilabel-classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dllllb/pytorch-lifestream/blob/master/tutorials/notebooks/multilabel-classification.ipynb)
- CoLES multimodal [notebook](tutorials/notebooks/CoLES-multimodal.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dllllb/pytorch-lifestream/blob/master/tutorials/notebooks/CoLES-multimodal.ipynb)

And we have a tutorials [here](tutorials)
## Docs

[Documentation](https://pytorch-lifestream.github.io/pytorch-lifestream/)

Library description [index](docs/index.md)

## Experiments on public datasets

`pytorch-lifestream` usage experiments on several public event datasets are available in the separate [repo](https://github.com/dllllb/ptls-experiments)

## PyTorch-LifeStream in ML Competitions
- [Data Fusion Contest 2022 report](https://habr.com/ru/companies/vtb/articles/673666/) (in Russian)
- [Data Fusion Contest 2022 report, Sber AI Lab team](https://habr.com/ru/companies/ods/articles/670572/) (in Russian)
- [VK.com Graph ML Hackaton report](https://habr.com/ru/companies/vk/articles/703484/) (in Russian)
- [VK.com Graph ML Hackaton report, AlfaBank team](https://habr.com/ru/companies/alfa/articles/698660/) (in Russian)
- [American Express - Default Prediction Kaggle contest report](https://habr.com/ru/articles/704440/) (in Russian)
- [Data Fusion Contest 2024, Sber AI Lab team](https://github.com/warofgam/Sber-AI-Lab---datafusion)
- [Data Fusion Contest 2024, Ivan Alexandrov](https://github.com/Ivanich-spb/datafusion_2024_churn) 
- [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)
    - [Supervised training with RNN](https://www.kaggle.com/code/ivkireev/amex-ptls-baseline-supervised-neural-network)
    - [Supervised training with Transformer](https://www.kaggle.com/code/ivkireev/amex-transformer-network-train-with-ptls)
    - [CoLES Embedding preparation](https://www.kaggle.com/code/ivkireev/amex-contrastive-embeddings-with-ptls-coles)
    - [CoLES Embedding usage as extra features for catboost](https://www.kaggle.com/code/ivkireev/catboost-classifier-with-coles-embeddings)
- [COTIC](https://github.com/VladislavZh/COTIC) - `pytorch-lifestream` is used in experiment for Continuous-time convolutions model of event sequences

## How to contribute

1. Make your chages via Fork and Pull request.
2. Write unit test for new code in `ptls_tests`.
3. Check unit test via `pytest`: [Example](.#install-from-source).

## Citation

We have a [paper](https://arxiv.org/abs/2002.08232) you can cite it:
```bibtex
@inproceedings{
   Babaev_2022, series={SIGMOD/PODS ’22},
   title={CoLES: Contrastive Learning for Event Sequences with Self-Supervision},
   url={http://dx.doi.org/10.1145/3514221.3526129},
   DOI={10.1145/3514221.3526129},
   booktitle={Proceedings of the 2022 International Conference on Management of Data},
   publisher={ACM},
   author={Babaev, Dmitrii and Ovsov, Nikita and Kireev, Ivan and Ivanova, Maria and Gusev, Gleb and Nazarov, Ivan and Tuzhilin, Alexander},
   year={2022},
   month=jun, collection={SIGMOD/PODS ’22}
}

```
