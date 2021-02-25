#!/usr/bin/env bash

spark-submit \
    --master local[8] \
    --name "Ephereum Make Dataset" \
    --driver-memory 16G \
    --conf spark.sql.shuffle.partitions=60 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="/mnt/ildar/spark_local_dir" \
    ../../make_datasets_spark.py \
    --data_path /mnt/ildar/ \
    --trx_files "/mnt/molchanov/etherium/out_txn.parquet" \
    --col_client_id "from_address" \
    --cols_event_time "dt_ts" \
    --cols_category "dt_event_weekday" "dt_event_hour" \
    --test_size 0.1 \
    --output_train_path "/mnt/ildar/ephereum_train_trx.parquet" \
    --output_test_path "/mnt/ildar/ephereum_trx.parquet" \
    --output_test_ids_path "/mnt/ildar/ephereum_test_ids.csv" \
    --log_file "results/dataset_gender.log"

# 152 sec with    --print_dataset_info
#  52 sec without --print_dataset_info
