# Get data

```sh
cd experiments/scenario_x5

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_x5
export SC_DEVICE="cuda"

# Prepare agg feature encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_params.json


# Train a supervised model and save scores to the file
python -m scenario_x5 fit_target params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_target_params.json


# Train the MeLES encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
# 
# 
# Fine tune the MeLES model in supervised mode and save scores to the file
# python -m scenario_gender fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json



# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_x5 compare_approaches --n_workers 2 \
    --baseline_name "agg_feat_embed.pickle" \
    --embedding_file_names "mles_embeddings.pickle" \
    --score_file_names "target_scores"


# check the results
cat results/scenario.csv

```
