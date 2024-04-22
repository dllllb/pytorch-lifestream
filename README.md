`PyTorch-LifeStream` or PTLS a library built upon [PyTorch](https://pytorch.org/) for building embeddings on discrete event sequences using self-supervision. It can process terabyte-size volumes of raw events like game history events, clickstream data, purchase history or card transactions.

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

pipenv sync  --dev # install packages exactly as specified in Pipfile.lock
pipenv shell
pytest

```

## Demo notebooks

Learn event sequence deep learning analysis with Pytorch-Lifestream.

We have collected a set of topics related to the processing of event sequences. Most themes are supported by demo code using the ptls library. We recommend studying the topics sequentially. However, if you are familiar in some areas, you can skip them and take only the relevant topics.

| ix   |  Topic                                  | Description                             | Demo  |
| ---- | --------------------------------------- | --------------------------------------- | ----- |
| 1.   | Prerequisites                           |                                         |       |
| 1.1. | torch                                   | Deep Learning framework                 | https://pytorch.org/       |
| 1.2. | pytorch-lightning                       | NN training framework                   | https://lightning.ai/      |
| 1.3. | (optional) hydra                        | Configuration framework                 | https://hydra.cc/ and [demo/Hydra CoLES Training.ipynb](./demo/Hydra CoLES Training.ipynb)         | 
| 1.4. | pandas                                  | Data preprocessing                      | https://pandas.pydata.org/ |
| 1.5. | (optional) spark                        | Big Data preprocessing                  | [https://spark.apache.org/](https://spark.apache.org/docs/latest/api/python/index.html) |
| 2.   | Event sequences                         | Problem statement and classical methods |     |
| 2.1. | Event sequence for global problems      | e.g. event sequence classification      | tbd |
| 2.2. | Event sequence for local problems       | e.g. next event prediction              | tbd |
| 3.     | Supervised neural networks              | Supervised learning for event sequence classification  | [demo/supervised-sequence-to-target.ipynb](./demo/su3ervised-sequence-to-target.ipynb)  |
| 3.1.   | Network Types                           | Different networks for sequences      |  |
| 3.1.1. | Recurrent neural networks               |    | tbd based on `supervised-sequence-to-target.ipynb` |
| 3.1.2. | (optional) Convolutional neural networks |    | tbd based on `supervised-sequence-to-target.ipynb` |
| 3.1.3. | Transformers                            |    | [demo/supervised-sequence-to-target-transformer.ipynb](demo/supervised-sequence-to-target-transformer.ipynb) |
| 3.2.   | Problem types                           | Different problems types for sequences  |  |
| 3.2.1. | Global problems                         | Binary, multilabel, regression, ...   | tbd based on [demo/multilabel-classification.ipynb](demo/multilabel-classification.ipynb) | 
| 3.2.2. | Local problems                          | Next event prediction                 | [demo/event-sequence-local-embeddings.ipynb](demo/event-sequence-local-embeddings.ipynb) |
| 4.   | Unsupervised learning                   | Pretrain self-supervised model with some proxy task | tbd based on [demo/coles-emb.ipynb](./demo/coles-emb.ipynb)  [![O4en In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dllllb/pytorch-lifestream/blob/master/demo/co4es-emb.ipynb)     |
| 4.1. | (optional) word2vec                     | Context based methods                   |     |
| 4.2. | mlm, rtd, gpt                           | Event bases methods                     | tbd |
| 4.3. | nsp, sop                                | Sequence based methods                  | [demo/nsp-sop-emb.ipynb](demo/nsp-sop-emb.ipynb) |
| 5.   | Contrastive and non-contrastive learning | Latent representation-based losses      | tbd based on [demo/coles-emb.ipynb](./demo/coles-emb.ipynb)             |
| 5.1. | coles                                   |    | [demo/coles-emb.ipynb](./demo/coles-emb.ipynb)                |
| 5.2. | vicreg                                  |    | tbd based on [demo/coles-emb.ipynb](./demo/coles-emb.ipynb)                   |
| 5.3. | cpc                                     |    | tbd based on [demo/coles-emb.ipynb](./demo/coles-emb.ipynb)                   |
| 5.4. | mlm, tabformer and others               | Self-supervised TrxEncoder only training with Masked Language Model | [demo/mlm-emb.ipynb](./demo/mlm-emb.ipynb) [demo/tabformer-emb.ipynb](demo/tabformer-emb.ipynb)                  |
| 6.   | Pretrained model usage                  |    |    |
| 6.1. | Downstream model on frozen embeddings   |    | tbd based on [demo/coles-emb.ipynb](./demo/coles-emb.ipynb)  |
| 6.2. | CatBoost embeddings features            |    | [demo/coles-catboost.ipynb](demo/coles-catboost.ipynb) |
| 6.3. | Model finetuning                        |    | [demo/coles-finetune.ipynb](./demo/coles-finetune.ipynb) |
| 7.   | Preprocessing options                   | Data preparation demos | [demo/preprocessing-demo.ipynb](demo/preprocessing-demo.ipynb) |
| 7.1  | ptls-format parquet data loading        | PySpark and Parquet for data preprocessing | [demo/pyspark-parquet.ipynb](demo/pyspark-parquet.ipynb) |
| 7.2. | Fast inference for big dataset          |    | [demo/extended_inference.ipynb](demo/extended_inference.ipynb) |
| 8.   | Features special types                  |    |    | 
| 8.1. | Using pretrained encoder to text features |  | [demo/coles-pretrained-embeddings.ipynb](demo/coles-pretrained-embeddings.ipynb) | 
| 8.2  | Multi source models                     |    | [demo/CoLES-demo-multimodal-unsupervised.ipynb](demo/CoLES-demo-multimodal-unsupervised.ipynb) |
| 9.   | Trx Encoding options                    |    |    | 
| 9.1. | Basic options                           |    | tbd | 
| 9.2. | Transaction Quantization                |    | tbd | 
| 9.3. | Transaction BPE                         |    | tbd | 

## Docs

[Documentation](https://dllllb.github.io/pytorch-lifestream/)

Library description [index](docs/index.md)

## Experiments on public datasets

`pytorch-lifestream` usage experiments on several public event datasets are available in the separate [repo](https://github.com/dllllb/ptls-experiments)

## PyTorch-LifeStream in ML competitions

- [Data Fusion Contest 2022 report](https://habr.com/ru/companies/vtb/articles/673666/) (in Russian)
- [Data Fusion Contest 2022 report, Sber AI Lab team](https://habr.com/ru/companies/ods/articles/670572/) (in Russian)
- [VK.com Graph ML Hackaton report](https://habr.com/ru/companies/vk/articles/703484/) (in Russian)
- [VK.com Graph ML Hackaton report, AlfaBank team](https://habr.com/ru/companies/alfa/articles/698660/) (in Russian)
- [American Express - Default Prediction Kaggle contest report](https://habr.com/ru/articles/704440/) (in Russian)

## How to contribute

1. Make your chages via Fork and Pull request.
2. Write unit test for new code in `ptls_tests`.
3. Check unit test via `pytest`: [Example](.#install-from-source).
