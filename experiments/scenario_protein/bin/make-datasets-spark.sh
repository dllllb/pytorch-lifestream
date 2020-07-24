#!/usr/bin/env bash

echo 'data prepare start...'
# python bin/data_prepare.py
echo 'data prepare complete'

spark-submit \
    --master local[12] \
    --name "Protein Make Dataset" \
    --driver-memory 100G \
    --conf spark.driver.memoryOverhead=100G \
    --conf spark.sql.shuffle.partitions=200 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    ../../make_datasets_spark.py \
    --data_path data/ \
    --trx_files "X_train.parquet" "X_test.parquet" \
    --col_client_id "uid" \
    --cols_event_time "#float" "pos" \
    --cols_category "amino_acid" \
    --cols_log_norm "amnt" \
    --target_files "target.csv" \
    --col_target "y" \
    --test_size "predefined" \
    --output_train_path "data/train_trx.parquet" \
    --output_test_path "data/test_trx.parquet" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_protein.log"

#  39 sec with    --print_dataset_info
#  3302 sec (0:55:03) sec without --print_dataset_info
