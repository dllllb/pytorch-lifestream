#!/usr/bin/env bash

export PYTHONPATH="../../"
SPARK_LOCAL_IP="127.0.0.1" spark-submit \
    --master local[24] \
    --name "Alpha battle Make Dataset" \
    --driver-memory 240G \
    --conf spark.sql.shuffle.partitions=500 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    make_dataset.py \
    --data_path './data/' \
    --col_client_id "app_id" \
    --cols_category "currency"	"operation_kind"	"card_type"	"operation_type" "operation_type_group" "ecommerce_flag" "payment_system" "income_flag" "mcc" "country" "city" "mcc_category" "day_of_week" "hour" "weekofyear" \
    --cols_log_norm "amnt" "hour_diff" "days_before" \
    --col_target "flag" \
    --test_size 0.1 \
    --output_train_path "data/train_trx.parquet" \
    --output_test_path "data/test_trx.parquet" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_alpha_battle.txt"

# ??? sec with
