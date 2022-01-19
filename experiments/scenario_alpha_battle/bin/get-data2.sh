#!/usr/bin/env bash

mkdir data

curl -OL https://boosters.pro/api/ch/files/pub/alfabattle2_test_transactions_contest.zip
curl -OL https://boosters.pro/api/ch/files/pub/alfabattle2_train_transactions_contest.zip
curl -OL https://boosters.pro/api/ch/files/pub/alfabattle2_alpha_sample.csv
curl -OL https://boosters.pro/api/ch/files/pub/alfabattle2_test_target_contest.csv
curl -OL https://boosters.pro/api/ch/files/pub/alfabattle2_train_target.csv

unzip alfabattle2_test_transactions_contest.zip -d data/
unzip alfabattle2_train_transactions_contest.zip -d data/
mv alfabattle2_test_transactions_contest.zip data/
mv alfabattle2_train_transactions_contest.zip data/
mv alfabattle2_*.csv data/

mv data/train_transactions_contest data/train_transactions_contest.parquet
mv data/test_transactions_contest data/test_transactions_contest.parquet


