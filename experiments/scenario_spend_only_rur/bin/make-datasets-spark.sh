#!/usr/bin/env bash

spark-submit \
    --master local[16] \
    --name "Age Make Dataset" \
    --driver-memory 200G \
    --conf spark.sql.shuffle.partitions=100 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    ../../make_datasets_spark.py \
    --data_path data/ \
    --trx_files transactions_train.csv transactions_test.csv \
    --col_client_id "client_id" \
    --cols_event_time "#float" "trans_date" \
    --cols_category "trans_date" "small_group" \
    --cols_log_norm "amount_rur" \
    --target_files train_target.csv \
    --col_target transac_total p_trans0 p_trans1 p_trans2 p_trans3 p_trans4 p_trans5 p_trans6 p_trans7 p_trans8 p_trans9 p_trans10 p_trans11 p_trans12 p_trans13 p_trans14 p_trans15 p_trans16 p_trans17 p_trans18 p_trans19 p_trans20 p_trans21 p_trans22 p_trans23 p_trans24 p_trans25 p_trans26 p_trans27 p_trans28 p_trans29 p_trans30 p_trans31 p_trans32 p_trans33 p_trans34 p_trans35 p_trans36 p_trans37 p_trans38 p_trans39 p_trans40 p_trans41 p_trans42 p_trans43 p_trans44 p_trans45 p_trans46 p_trans47 p_trans48 p_trans49 p_trans50 p_trans51 \
    --test_size 0.1 \
    --output_train_path "data/train_trx.parquet" \
    --output_test_path "data/test_trx.parquet" \
    --output_test_ids_path "data/kruzhilov/data/test_ids.csv" \
    --log_file "results/dataset_age_pred.log"

# 654 sec with    --print_dataset_info
# 144 sec without --print_dataset_info
