#!/usr/bin/env bash

mkdir data

curl -OL https://storage.yandexcloud.net/di-datasets/trans-gender-2019.zip
unzip trans-gender-2019.zip -d data
mv trans-gender-2019.zip data/
