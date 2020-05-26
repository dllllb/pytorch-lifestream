#!/usr/bin/env bash

mkdir data

curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/9c6913e5/retailhero-uplift.zip
unzip -j retailhero-uplift.zip 'data/*' -d data
mv retailhero-uplift.zip data/
