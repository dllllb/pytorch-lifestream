# Get data

```sh
cd experiments/scenario_age_pred

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make_datasets_spark_file.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

sh bin/run_all_scenarios.sh
```

### Semi-supervised setup
```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

# run semi supervised scenario
./bin/scenario_semi_supervised.sh

# check the results
cat results/semi_scenario_age_pred_*.csv
```

### Test model configurations

```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

# run all scenarios or select one
./bin/*.sh

# check the results
cat results/scenario_age_pred_*.csv
```

### Transformer network
```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

# Train the MeLES encoder on transformer and take embedidngs; inference
python ../../pl_train_module.py  --conf conf/transformer_params.hocon
python ../../pl_inference.py --conf conf/transformer_params.hocon

python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_on_transf_params.hocon

# Check some options with `--help` argument
python -m scenario_age_pred compare_approaches --n_workers 1 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "transf_embeddings.pickle" \
    --score_file_names "transf_finetuning_scores"
```

### Projection head network (like SimCLR)
```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

# Train the encoder on transformer and take embedidngs; inference
python ../../pl_train_module.py --conf conf/mles_proj_head_params.hocon
python ../../pl_inference.py --conf conf/mles_proj_head_params.hocon

# Check some options with `--help` argument
python -m scenario_age_pred compare_approaches --n_workers 3 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "mles_proj_head_embeddings.pickle"
```

# CPC v2 [TODO]

```sh
# Train the Contrastive Predictive Coding (CPC v2) model; inference 
python ../../cpc_v2_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_v2_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_v2_params.json


# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_age_pred compare_approaches --n_workers 1 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "cpc_v2_embeddings.pickle"

```

# Complex Learning [TODO]

```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"

# Train complex model and get an embeddings
python ../../complex_learning.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/complex_learning_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/complex_learning_params.json

python -m scenario_age_pred compare_approaches --n_workers 5 \
    --output_file "results/scenario_age_pred__complex_learning.csv" \
    --embedding_file_names "complex_embeddings.pickle"
```

# Distribution targets scenario

```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0  # define here one gpu device number

[ -d data/ ] && rm -r data/
[ -d conf/embeddings_validation.work/ ] && rm -r conf/embeddings_validation.work/
[ -d lightning_logs/ ] && rm -r lightning_logs/

bin/get-data.sh
python distribution_target.py
bin/make_datasets_spark_file.sh

python -m embeddings_validation \
    --conf conf/embeddings_validation_distribution_target.hocon --workers 10 --total_cpu_count 20 --split_only --local_scheduler

python ../../pl_fit_target.py --conf conf/pl_fit_distribution_target.hocon

python ../../pl_fit_target.py --conf conf/pl_fit_distribution_target_statistics.hocon

python -m embeddings_validation \
    --conf conf/embeddings_validation_distribution_target.hocon --workers 10 --total_cpu_count 20 --local_scheduler

```
