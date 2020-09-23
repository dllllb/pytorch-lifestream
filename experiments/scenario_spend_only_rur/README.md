# Get data

```sh
cd experiments/scenario_spend_only_rur
# download datasets
bin/get-data.sh

#create ground truth - a share of each transaction type for a client
python transac2statistics.py
# convert datasets from transaction list to features for metric learning
bin/make-datasets-spark.sh
#if the previous command does not work, set - export SPARK_LOCAL_IP="127.0.0.1" before 

```
```sh
# Main scenario
export SC_DEVICE="cuda" #"cuda:1" if cuda:0 is busy

# Train the MeLES encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json

# train a network to predict transaction shares from client embeddings
python -m scenario_spend_only_rur fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json

python -m scenario_spend_only_rur compare_approaches --n_workers 1 --score_file_names "mles_finetuning_scores" 

# check the results
cat results/scenario.csv
```

