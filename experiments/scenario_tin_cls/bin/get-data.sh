#!/usr/bin/env bash

mkdir data

curl -OL https://storage.googleapis.com/di-datasets/data-like-tinkoff-2019.zip
unzip data-like-tinkoff-2019.zip -d data
chmod +r data/*
mv data-like-tinkoff-2019.zip data/
