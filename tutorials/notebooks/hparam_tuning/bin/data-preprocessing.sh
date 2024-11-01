#!/usr/bin/env bash

spark-submit \
    --master local[8] \
    --name "Gender Make Dataset" \
    --driver-memory 16G \
    --conf spark.driver.memoryOverhead="4g" \
    --conf spark.local.dir="data/.spark_local_dir" \
    data_preprocessing.py
