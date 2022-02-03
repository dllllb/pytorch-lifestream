`pytorch-lifestream` a library built upon [PyTorch](https://pytorch.org/) for building embeddings on discrete event sequences using self-supervision. It can process terabyte-size volumes of raw events like game history events, clickstream data, purchase history or card transactions.

It supports various methods of self-supervised training, adapted for event sequences:

- Contrastive Learning for Event Sequences ([CoLES](https://arxiv.org/abs/2002.08232))
- Contrastive Predictive Coding ([CPC](https://arxiv.org/abs/1807.03748))
- Replaced Token Detection (RTD) from [ELECTRA](https://arxiv.org/abs/2003.10555)
- Next Sequence Prediction (NSP) from [BERT](https://arxiv.org/abs/1810.04805)
- Sequences Order Prediction (SOP) from [ALBERT](https://arxiv.org/abs/1909.11942)

It supports several types of encoders, including Transformer and RNN. It also supports many types of self-supervised losses.

The following variants of the contrastive losses are supported:

- Contrastive loss ([paper](https://doi.org/10.1109/CVPR.2006.100))
- Triplet loss ([paper](https://arxiv.org/abs/1412.6622))
- Binomial deviance loss ([paper](https://arxiv.org/abs/1407.4979))
- Histogramm loss ([paper](https://arxiv.org/abs/1611.00822))
- Margin loss ([paper](https://arxiv.org/abs/1706.07567))

## Install from PyPi

```sh
pip install pytorch-lifestream
```

## Install from source

```sh
# Ubuntu 20.04

sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv sync  --dev # install packages exactly as specified in Pipfile.lock
pipenv shell
pytest

```

## Demo example

Demo example can be found in the [notebook](demo/example.ipynb)
