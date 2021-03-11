#!/usr/bin/env bash

export PYTHONPATH="../../"
spark-submit \
    --master local[8] \
    --name "Blow 2019 Make Dataset" \
    --driver-memory 54G \
    --conf spark.sql.shuffle.partitions=60 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    make_dataset.py \
    --data_path data/ \
    --trx_files "train.csv" "test.csv" \
    --col_client_id "game_session" \
    --cols_event_time "timestamp" \
    --cols_category "event_id" "event_code" "event_type" "title" "world" "correct" \
    --target_files "train_labels.csv" \
    --col_target "accuracy_group" \
    --max_trx_count=7000 \
    --test_size 0.1 \
    --output_train_path "data/train_trx.parquet" \
    --output_test_path "data/test_trx.parquet" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_bowl2019.log"

    # --print_dataset_info
    # --cols_log_norm "amount"
