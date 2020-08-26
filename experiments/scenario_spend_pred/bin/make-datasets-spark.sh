#!/usr/bin/env bash

spark-submit \
    --master local[16] \
    --name "Age Make Dataset" \
    --driver-memory 80G \
    --conf spark.sql.shuffle.partitions=500 \
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
    --col_target transac_total p_trans0 p_trans1 p_trans2 p_trans3 p_trans4 p_trans5 p_trans6 p_trans7 p_trans8 p_trans9 p_trans10 p_trans11 p_trans12 p_trans13 p_trans14 p_trans15 p_trans16 p_trans17 p_trans18 p_trans19 p_trans20 p_trans21 p_trans22 p_trans23 p_trans24 p_trans25 p_trans26 p_trans27 p_trans28 p_trans29 p_trans30 p_trans31 p_trans32 p_trans33 p_trans34 p_trans35 p_trans36 p_trans37 p_trans38 p_trans39 p_trans40 p_trans41 p_trans42 p_trans43 p_trans44 p_trans45 p_trans46 p_trans47 p_trans48 p_trans49 p_trans50 p_trans51 mean_rur0 mean_rur1 mean_rur2 mean_rur3 mean_rur4 mean_rur5 mean_rur6 mean_rur7 mean_rur8 mean_rur9 mean_rur10 mean_rur11 mean_rur12 mean_rur13 mean_rur14 mean_rur15 mean_rur16 mean_rur17 mean_rur18 mean_rur19 mean_rur20 mean_rur21 mean_rur22 mean_rur23 mean_rur24 mean_rur25 mean_rur26 mean_rur27 mean_rur28 mean_rur29 mean_rur30 mean_rur31 mean_rur32 mean_rur33 mean_rur34 mean_rur35 mean_rur36 mean_rur37 mean_rur38 mean_rur39 mean_rur40 mean_rur41 mean_rur42 mean_rur43 mean_rur44 mean_rur45 mean_rur46 mean_rur47 mean_rur48 mean_rur49 mean_rur50 mean_rur51\
    --test_size 0.1 \
    --output_train_path "data/train_trx.parquet" \
    --output_test_path "data/test_trx.parquet" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_age_pred.log"

# 654 sec with    --print_dataset_info
# 144 sec without --print_dataset_info
