# Get data

```sh
cd experiments/scenario_protein

# Check your ~/.kaggle/kaggle.json before data download
# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets-spark.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_protein
export SC_DEVICE="cuda"

# Prepare agg feature encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_params.json

# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_protein compare_approaches --n_workers 1 \
    --models lgb \
    --baseline_name "agg_feat_embed.pickle" 

# check the results
cat results/scenario.csv
```
