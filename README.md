![ptls-logo](https://github.com/Dzagcoffee/pytorch-lifestream/assets/109000119/1dd45ef8-2a9e-4e4e-b338-beb8a08069e6)

[![GitHub license](https://img.shields.io/github/license/dllllb/pytorch-lifestream.svg)](https://github.com/dllllb/pytorch-lifestream/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/pytorch-lifestream.svg)](https://badge.fury.io/py/pytorch-lifestream)
[![GitHub issues](https://img.shields.io/github/issues/dllllb/pytorch-lifestream.svg)](https://github.com/dllllb/pytorch-lifestream/issues)
[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/pytorch_lifestream)

`pytorch-lifestream` a library built upon [PyTorch](https://pytorch.org/) for building embeddings on discrete event sequences using self-supervision. It can process terabyte-size volumes of raw events like game history events, clickstream data, purchase history or card transactions.

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

- Supervised model training [notebook](demo/supervised-sequence-to-target.ipynb)
- Self-supervided training and embeddings for downstream task [notebook](demo/coles-emb.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dllllb/pytorch-lifestream/blob/master/demo/coles-emb.ipynb)
- Self-supervided embeddings in CatBoost [notebook](demo/coles-catboost.ipynb)
- Self-supervided training and fine-tuning [notebook](demo/coles-finetune.ipynb)
- Self-supervised TrxEncoder only training with Masked Language Model task and fine-tuning [notebook](demo/mlm-emb.ipynb)
- Pandas data preprocessing options [notebook](demo/preprocessing-demo.ipynb)
- PySpark and Parquet for data preprocessing [notebook](demo/pyspark-parquet.ipynb)
- Fast inference on large dataset [notebook](demo/extended_inference.ipynb)
- Supervised multilabel classification [notebook](demo/multilabel-classification.ipynb)

## Docs

[Documentation](https://dllllb.github.io/pytorch-lifestream/)

Library description [index](docs/index.md)

## Experiments on public datasets

`pytorch-lifestream` usage experiments on several public event datasets are available in the separate [repo](https://github.com/dllllb/ptls-experiments)

## Citation

We have a [paper](https://arxiv.org/abs/2002.08232) you can cite it:
```bibtex
@inproceedings{Babaev_2022, series={SIGMOD/PODS ’22},
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
