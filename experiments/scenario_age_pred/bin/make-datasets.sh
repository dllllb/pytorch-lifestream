#!/usr/bin/env bash

python ../../make_datasets.py \
    --data_path data/ \
    --trx_files transactions_train.csv transactions_test.csv \
    --col_client_id "client_id" \
    --cols_event_time "#float" "trans_date" \
    --cols_category "trans_date" "small_group" \
    --cols_log_norm "amount_rur" \
    --target_files train_target.csv \
    --col_target bins \
    --test_size 0.1 \
    --output_train_path "data/train_trx.p" \
    --output_test_path "data/test_trx.p" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_age_pred.log"
