#!/usr/bin/env bash

mkdir data

kaggle datasets download -d igempotsdam/protein-heat-resistance-dataset

unzip protein-heat-resistance-dataset.zip -d data
mv protein-heat-resistance-dataset.zip data/
