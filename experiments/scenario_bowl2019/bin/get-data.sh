#!/usr/bin/env bash

mkdir data

wget https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/31erlljbW6Px9A -O data-science-bowl-2019.zip
unzip -j data-science-bowl-2019.zip -d data
mv data-science-bowl-2019.zip data/