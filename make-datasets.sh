#!/usr/bin/env bash

python opends/make_datasets.py \
    --data_path data/age-pred/ \
    --trx_files transactions_train.csv transactions_test.csv \
    --col_client_id "client_id" \
    --cols_event_time "trans_date" \
    --cols_category "trans_date" "small_group" \
    --cols_log_norm "amount_rur" \
    --output_path "data/age-pred/all_trx.p"

python opends/make_datasets.py \
    --data_path data/tinkoff/ \
    --trx_files transactions.csv \
    --col_client_id "customer_id" \
    --cols_event_time "transaction_month" "transaction_day" \
    --cols_category "transaction_month" "transaction_day" "merchant_id" "merchant_mcc" \
    --cols_log_norm "transaction_amt" \
    --output_path "data/tinkoff/all_trx.p"

python opends/make_datasets.py \
    --data_path data/gender/ \
    --trx_files transactions.csv \
    --col_client_id "customer_id" \
    --cols_event_time "tr_datetime" \
    --cols_category "mcc_code" "tr_type" "term_id" \
    --cols_log_norm "amount" \
    --output_path "data/gender/all_trx.p"