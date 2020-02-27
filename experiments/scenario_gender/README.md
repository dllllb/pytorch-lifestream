# Get data

```sh
cd experiments/scenario_gender

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_gender
export SC_DEVICE="cuda"

# Train a supervised model and save scores to the file
python -m scenario_gender fit_target params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_target_params.json


# Train the MeLES encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
# Fine tune the MeLES model in supervised mode and save scores to the file
python -m scenario_gender fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json


# Train the Contrastive Predictive Coding (CPC) model; inference
python ../../train_cpc.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json
# Fine tune the CPC model in supervised mode and save scores to the file
python -m scenario_gender fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_cpc_params.json


# Train the Contrastive Predictive Coding (CPC v2) model; inference 
python ../../cpc_v2_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_v2_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_v2_params.json


# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_gender compare_approaches --n_workers 1 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "mles_embeddings.pickle" "cpc_embeddings.pickle" "cpc_v2_embeddings.pickle" \
    --score_file_names "target_scores" "mles_finetuning_scores" 'cpc_finetuning_scores'


# check the results
cat results/scenario.csv
```

# Semi-supervised setup
```sh
cd experiments/scenario_gender
export SC_DEVICE="cuda"

# run semi supervised scenario
./bin/scenario_semi_supervised.sh

# check the results
cat runs/semi_scenario_gender_*.csv

```

# Test model configurations

```sh
cd experiments/scenario_gender
export SC_DEVICE="cuda"

# run all scenarios or select one
./bin/*.sh

# check the results
cat runs/scenario_gender_*.csv
```

### Transformer network
```sh
cd experiments/gender
export SC_DEVICE="cuda"

# Train the MeLES encoder on transformer and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/transformer_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/transformer_params.json

python -m scenario_gender fit_finetuning \
    params.device="$SC_DEVICE" \
    --conf conf/dataset.hocon conf/transformer_finetuning.json

# Check some options with `--help` argument
python -m scenario_gender compare_approaches --n_workers 3 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "transf_embeddings.pickle" \
    --score_file_names "transf_finetuning_scores"

```
