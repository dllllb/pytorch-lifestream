# Get data

```sh
cd experiments/scenario_gender

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets-spark.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_gender
export SC_DEVICE="cuda"


sh bin/run_all_scenarios.sh

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

### Projection head network (like SimCLR)
```sh
cd experiments/gender
export SC_DEVICE="cuda"

# Train the encoder on transformer and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_proj_head_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_proj_head_params.json

# Check some options with `--help` argument
python -m scenario_gender compare_approaches --n_workers 3 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "mles_proj_head_embeddings.pickle"

```

### CPC v2
```sh
cd experiments/gender
export SC_DEVICE="cuda"

# Train the Contrastive Predictive Coding (CPC v2) model; inference 
python ../../cpc_v2_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_v2_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_v2_params.json

# Check some options with `--help` argument
python -m scenario_gender compare_approaches --n_workers 3 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "cpc_v2_embeddings.pickle"
```

# pytorch_lightning framework
```sh
#  `conf/dataset_iterable_file.hocon` may be included in `conf/mles_params.hocon`
python ../../pl_train_coles.py \
     trainer.gpus=[3] \
     --conf conf/dataset_iterable_file.hocon conf/mles_params.hocon
python ../../pl_inference.py \
    params.device="cuda:3" \
    --conf conf/mles_params.hocon
```
