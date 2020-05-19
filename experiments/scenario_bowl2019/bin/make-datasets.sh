#!/usr/bin/env bash

python ../scenario_bowl2019/make_datasets.py \
    --data_path data/ \
    --trx_files train.csv test.csv \
    --col_client_id "installation_id" \
    --col_sample_id "game_session" \
    --cols_event_time "#datetime_string" "timestamp" \
    --cols_event_data_parse "correct" \
    --cols_category "event_id" "event_code" "event_type" "title" "world" "correct" \
    --cols_identity "game_time" \
    --cols_enumerate "game_session" \
    --target_files train_labels.csv \
    --test_size 0.1 \
    --output_train_trx_path "data/train_trx.p" \
    --output_train_mapping_path "data/encoding.p" \
    --output_train_path "data/train.p" \
    --output_test_path "data/test.p" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_bowl2019.log"