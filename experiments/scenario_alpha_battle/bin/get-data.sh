#!/usr/bin/env bash

# the same as get_data2.sh
mkdir data

curl -OL https://storage.googleapis.com/di-datasets/alfabattle2b-boosters.pro.zip

unzip alfabattle2b-boosters.pro.zip -d data/
mv alfabattle2b-boosters.pro.zip data/

mv data/train_transactions_contest data/train_transactions_contest.parquet
mv data/test_transactions_contest data/test_transactions_contest.parquet
