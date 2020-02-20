#!/usr/bin/env bash

python ../../make_datasets.py \
    --data_path data/ \
    --trx_files transactions.csv \
    --col_client_id "customer_id" \
    --cols_event_time "transaction_month" "transaction_day" \
    --cols_category "transaction_month" "transaction_day" "merchant_id" "merchant_mcc" \
    --cols_log_norm "transaction_amt" \
    --target_files "stories_reaction_train.csv" \
    --test_size 0.1 \
    --output_train_path "data/train_trx.p" \
    --output_test_path "data/test_trx.p" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_tinkoff.log"
