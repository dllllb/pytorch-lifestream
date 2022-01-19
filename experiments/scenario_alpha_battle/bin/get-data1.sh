#!/usr/bin/env bash

mkdir data

curl -OL https://boosters.pro/api/ch/files/pub/alfabattle2_abattle_clickstream.zip
curl -OL https://boosters.pro/api/ch/files/pub/alfabattle2_abattle_sample_prediction.csv
curl -OL https://boosters.pro/api/ch/files/pub/alfabattle2_abattle_train_target.csv
curl -OL https://boosters.pro/api/ch/files/pub/alfabattle2_prediction_session_timestamp.csv

unzip alfabattle2_abattle_clickstream.zip -d data/clickstream/
mv alfabattle2_abattle_clickstream.zip data/
mv alfabattle2_*.csv data/

