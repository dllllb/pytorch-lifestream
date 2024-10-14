#!/usr/bin/env bash

git clone --depth 1 https://huggingface.co/datasets/dllllb/alfa-scoring-trx data

gunzip -f data/*.csv.gz

mv data/train_transactions data/train_transactions_contest.parquet
mv data/test_transactions data/test_transactions_contest.parquet
