#!/usr/bin/env bash

spark-submit \
    --master local[16] \
    --name "X5 Make Dataset" \
    --driver-memory 200G \
    --conf spark.sql.shuffle.partitions=100 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    ../../make_datasets_spark.py \
    --data_path data/ \
    --trx_files purchases.csv \
    --dict products.csv product_id \
    --col_client_id "client_id" \
    --cols_event_time "#datetime" "transaction_datetime" \
    --cols_category store_id product_id level_1 level_2 level_3 level_4 segment_id brand_id vendor_id is_own_trademark is_alcohol \
    --cols_log_norm netto regular_points_received express_points_received regular_points_spent express_points_spent purchase_sum product_quantity trn_sum_from_iss trn_sum_from_red \
    --target_files clients.csv \
    --col_target age gender \
    --test_size 0.1 \
    --output_train_path "data/train_trx.parquet" \
    --output_test_path "data/test_trx.parquet" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_x5.log"

#  4491 sec (1:14:51) with    --print_dataset_info
#  sec without --print_dataset_info
