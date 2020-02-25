#!/usr/bin/env bash

python ../../make_datasets.py \
    --data_path data/ \
    --trx_files transactions.csv \
    --col_client_id "customer_id" \
    --cols_event_time "tr_datetime" \
    --cols_category "mcc_code" "tr_type" "term_id" \
    --cols_log_norm "amount" \
    --target_files gender_train.csv \
    --col_target gender \
    --test_size 0.1 \
    --output_train_path "data/train_trx.p" \
    --output_test_path "data/test_trx.p" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_gender.log"
