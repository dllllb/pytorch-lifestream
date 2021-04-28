#!/usr/bin/env bash

spark-submit \
    --master local[8] \
    --name "Gender Make Dataset" \
    --driver-memory 16G \
    --conf spark.sql.shuffle.partitions=60 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    ../../make_datasets_spark.py \
    --data_path data/ \
    --trx_files transactions.csv \
    --col_client_id "customer_id" \
    --cols_event_time "#gender" "tr_datetime" \
    --cols_category "mcc_code" "tr_type" "term_id" \
    --cols_log_norm "amount" \
    --target_files gender_train.csv \
    --col_target gender \
    --test_size 0.1 \
    --output_train_path "data/train_trx.parquet" \
    --output_test_path "data/test_trx.parquet" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_gender.log" \
    --print_dataset_info

# 152 sec with    --print_dataset_info
#  52 sec without --print_dataset_info
