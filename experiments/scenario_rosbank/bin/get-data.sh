#!/usr/bin/env bash

mkdir data

curl -OL https://storage.googleapis.com/di-datasets/rosbank-ml-contest-boosters.pro.zip
unzip rosbank-ml-contest-boosters.pro.zip -d data
mv rosbank-ml-contest-boosters.pro.zip data/
