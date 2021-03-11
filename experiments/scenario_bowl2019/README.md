# Get data

```sh
cd experiments/scenario_bowl2019

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets-spark.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_bowl2019
export SC_DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

# Train a supervised model and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_target.hocon

# Train the MeLES encoder and take embedidngs; inference
python ../../pl_train_module.py --conf conf/mles_params.hocon
python ../../pl_inference.py --conf conf/mles_params.hocon

# Fine tune the MeLES model in supervised mode and save scores to the file

python ../../pl_train_module.py params.train.neg_count=5 model_path="models/mles_model_ft.p" --conf conf/mles_params.hocon

python ../../pl_fit_target.py params.pretrained.model_path="models/mles_model_ft.p" data_module.train.drop_last=true --conf conf/pl_fit_finetuning_mles.hocon

# Train the Contrastive Predictive Coding (CPC) model; inference
python ../../pl_train_module.py --conf conf/cpc_params.hocon
python ../../pl_inference.py --conf conf/cpc_params.hocon

# Train a special CPC model for fine-tuning 
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model. 
python ../../pl_train_module.py --conf conf/cpc_params_for_finetuning.hocon

# Fine tune the CPC model in supervised mode and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_cpc.hocon

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
export CUDA_VISIBLE_DEVICES=0

# run semi supervised scenario
./bin/scenario_semi_supervised.sh

# check the results
cat results/scenario_bowl2019_*.txt
```

### Test model configurations

```sh
cd experiments/scenario_bowl2019
export SC_DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

# run all scenarios or select one
./bin/scenario*.sh

# check the results
cat results/scenario_bowl2019_*.csv
```

# Complex Learning [TODO]

```sh
cd experiments/scenario_bowl2019
export SC_DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

# Train complex model and get an embeddings
python ../../complex_learning.py    params.device="$SC_DEVICE" --conf conf/trx_dataset.hocon conf/complex_learning_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/complex_learning_params.json

python -m scenario_bowl2019 compare_approaches --n_workers 5 \
    --output_file "results/scenario_bowl2019__complex_learning.csv" \
    --embedding_file_names "complex_embeddings.pickle"

```
