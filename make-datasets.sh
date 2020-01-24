#!/usr/bin/env bash

python opends/make_datasets.py \
    --data_path data/age-pred/ \
    --trx_files transactions_train.csv transactions_test.csv \
    --col_client_id "client_id" \
    --cols_event_time "trans_date" \
    --cols_category "trans_date" "small_group" \
    --cols_log_norm "amount_rur" \
    --target_files train_target.csv \
    --col_target bins \
    --output_train_path "data/age-pred/train_trx.p" \
    --output_test_path "data/age-pred/test_trx.p" \
    --output_test_ids_path "data/age-pred/test_ids.csv" \
    --log_file "opends/runs/dataset_age_pred.log"

python opends/make_datasets.py \
    --data_path data/tinkoff/ \
    --trx_files transactions.csv \
    --col_client_id "customer_id" \
    --cols_event_time "transaction_month" "transaction_day" \
    --cols_category "transaction_month" "transaction_day" "merchant_id" "merchant_mcc" \
    --cols_log_norm "transaction_amt" \
    --target_files "stories_reaction_train.csv" \
    --test_size 0.0 \
    --output_train_path "data/tinkoff/train_trx.p" \
    --output_test_path "data/tinkoff/test_trx.p" \
    --output_test_ids_path "data/tinkoff/test_ids.csv" \
    --log_file "opends/runs/dataset_tinkoff.log"

python opends/make_datasets.py \
    --data_path data/gender/ \
    --trx_files transactions.csv \
    --col_client_id "customer_id" \
    --cols_event_time "tr_datetime" \
    --cols_category "mcc_code" "tr_type" "term_id" \
    --cols_log_norm "amount" \
    --target_files gender_train.csv \
    --col_target gender \
    --test_size 0.1 \
    --output_train_path "data/gender/train_trx.p" \
    --output_test_path "data/gender/test_trx.p" \
    --output_test_ids_path "data/gender/test_ids.csv" \
    --log_file "opends/runs/dataset_gender.log"
