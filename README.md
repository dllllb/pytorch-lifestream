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

- Supervised model training [notebook](tutorials/notebooks/supervised-sequence-to-target.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dllllb/pytorch-lifestream/blob/master/demo/coles-emb.ipynb)
- Self-supervided training and embeddings for downstream task [notebook](tutorials/notebooks/coles-emb.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dllllb/pytorch-lifestream/blob/master/demo/coles-emb.ipynb)
- Self-supervided embeddings in CatBoost [notebook](tutorials/notebooks/coles-catboost.ipynb)
- Self-supervided training and fine-tuning [notebook](tutorials/notebooks/coles-finetune.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xu5hjYJRtSvu12haMnbR4KtGsNkk4cnv#scrollTo=WyOYsMF2SEZ3)
- Self-supervised TrxEncoder only training with Masked Language Model task and fine-tuning [notebook](tutorials/notebooks/mlm-emb.ipynb)
- Pandas data preprocessing options [notebook](tutorials/notebooks/preprocessing-demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wwWl5hhdCdOTa6aBS4sSpTD8kF0NZQzA?usp=sharing)
- PySpark and Parquet for data preprocessing [notebook](tutorials/notebooks/pyspark-parquet.ipynb)
- Fast inference on large dataset [notebook](tutorials/notebooks/extended_inference.ipynb)
- Supervised multilabel classification [notebook](tutorials/notebooks/coles-emb.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bj5pDXd-XHJUKSqWz4bmwPsAi9M8L5wq)
- CoLES multimodal [notebook](tutorials/notebooks/coles-emb.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oV18ehvyPhhPjtolx8qrWw5ojreVJL_c)

And we have a tutorials [here](tutorials)
## Docs

[Documentation](https://dllllb.github.io/pytorch-lifestream/)

Library description [index](docs/index.md)

## Experiments on public datasets

`pytorch-lifestream` usage experiments on several public event datasets are available in the separate [repo](https://github.com/dllllb/ptls-experiments)

## PyTorch-LifeStream in ML Competitions

### 📄 Reports (Habr):
- [![Habr](https://img.shields.io/badge/Habr-Data%20Fusion%20Contest%202022-blue)](https://habr.com/ru/companies/vtb/articles/673666/)
- [![Habr](https://img.shields.io/badge/Habr-Sber%20AI%20Lab%20team%20Data%20Fusion%202022-blue)](https://habr.com/ru/companies/ods/articles/670572/)
- [![Habr](https://img.shields.io/badge/Habr-VK.com%20Graph%20ML%20Hackathon-blue)](https://habr.com/ru/companies/vk/articles/703484/)
- [![Habr](https://img.shields.io/badge/Habr-AlfaBank%20VK.com%20Graph%20ML%20Hackathon-blue)](https://habr.com/ru/companies/alfa/articles/698660/)
- [![Habr](https://img.shields.io/badge/Habr-American%20Express%20Kaggle%20Contest-blue)](https://habr.com/ru/articles/704440/)

### 💻 Code (GitHub/Kaggle):
- [![GitHub](https://img.shields.io/badge/GitHub-Sber%20AI%20Lab%20Data%20Fusion%202024-green)](https://github.com/warofgam/Sber-AI-Lab---datafusion)
- [![GitHub](https://img.shields.io/badge/GitHub-Ivan%20Alexandrov%20Data%20Fusion%202024-green)](https://github.com/Ivanich-spb/datafusion_2024_churn)
- [![Kaggle](https://img.shields.io/badge/Kaggle-American%20Express%20Kaggle%20Contest-blue)](https://www.kaggle.com/competitions/amex-default-prediction)
  - [![Kaggle](https://img.shields.io/badge/Kaggle-RNN%20Supervised%20Training-blue)](https://www.kaggle.com/code/ivkireev/amex-ptls-baseline-supervised-neural-network)
  - [![Kaggle](https://img.shields.io/badge/Kaggle-Transformer%20Supervised%20Training-blue)](https://www.kaggle.com/code/ivkireev/amex-transformer-network-train-with-ptls)
  - [![Kaggle](https://img.shields.io/badge/Kaggle-CoLES%20Embedding%20Preparation-blue)](https://www.kaggle.com/code/ivkireev/amex-contrastive-embeddings-with-ptls-coles)
  - [![Kaggle](https://img.shields.io/badge/Kaggle-CatBoost%20with%20CoLES%20Embeddings-blue)](https://www.kaggle.com/code/ivkireev/catboost-classifier-with-coles-embeddings)
- [![GitHub](https://img.shields.io/badge/GitHub-COTIC%20Event%20Sequences%20Model-green)](https://github.com/VladislavZh/COTIC)
  
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
