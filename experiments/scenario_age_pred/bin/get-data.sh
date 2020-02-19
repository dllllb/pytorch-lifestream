#!/usr/bin/env bash

mkdir data
cd data
curl -OL https://storage.googleapis.com/di-datasets/age-prediction-nti-sbebank-2019.zip
unzip -j age-prediction-nti-sbebank-2019.zip 'data/*.csv' -d age-pred

curl -OL https://storage.googleapis.com/di-datasets/data-like-tinkoff-2019.zip
unzip data-like-tinkoff-2019.zip -d tinkoff
chmod +r tinkoff/*

curl -OL https://storage.googleapis.com/di-datasets/trans-gender-2019.zip
unzip trans-gender-2019.zip -d gender
