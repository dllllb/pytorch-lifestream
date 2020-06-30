# Get data

```sh
cd experiments/scenario_bowl2019

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_bowl2019
export SC_DEVICE="cuda"

# Train a supervised model and save scores to the file
python -m scenario_bowl2019 fit_target params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_target_params.json

# Train the MeLES encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/trx_dataset.hocon conf/mles_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json

# Fine tune the MeLES model in supervised mode and save scores to the file
python -m scenario_bowl2019 fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json

# Train the Contrastive Predictive Coding (CPC) model; inference
python ../../train_cpc.py    params.device="$SC_DEVICE" --conf conf/trx_dataset.hocon conf/cpc_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json

# Train a special CPC model for fine-tuning 
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model. 
python ../../train_cpc.py params.device="$SC_DEVICE" --conf conf/trx_dataset.hocon conf/cpc_params_for_finetuning.json
# Fine tune the CPC model in supervised mode and save scores to the file
python -m scenario_bowl2019 fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_cpc_params.json

# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_bowl2019 compare_approaches --n_workers 1 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "mles_embeddings.pickle" "cpc_embeddings.pickle" \
    --score_file_names "target_scores" "mles_finetuning_scores" 'cpc_finetuning_scores'


# check the results
cat results/scenario.csv
```

### Semi-supervised setup
```sh
cd experiments/scenario_bowl2019
export SC_DEVICE="cuda"

# run semi supervised scenario
./bin/scenario_semi_supervised.sh

# check the results
cat results/semi_scenario_bowl2019_*.csv

```

### Test model configurations

```sh
cd experiments/scenario_bowl2019
export SC_DEVICE="cuda"

# run all scenarios or select one
./bin/scenario*.sh

# check the results
cat results/scenario_bowl2019_*.csv
```

# Complex Learning

```sh
cd experiments/scenario_bowl2019
export SC_DEVICE="cuda"

# Train complex model and get an embeddings
python ../../complex_learning.py    params.device="$SC_DEVICE" --conf conf/trx_dataset.hocon conf/complex_learning_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/complex_learning_params.json

python -m scenario_bowl2019 compare_approaches --n_workers 5 \
    --output_file "results/scenario_bowl2019__complex_learning.csv" \
    --embedding_file_names "complex_embeddings.pickle"

```
