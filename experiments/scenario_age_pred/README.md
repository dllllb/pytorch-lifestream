# Get data

```sh
cd experiments/scenario_age_pred

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets-spark.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"

# Train a supervised model and save scores to the file
python -m scenario_age_pred fit_target params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_target_params.json


# Train the MeLES encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json


# Train a special MeLES model for fine-tuning 
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model. 
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params_for_finetuning.json
# Take the pretrained metric learning model and fine tune it in supervised mode; save scores to the file
python -m scenario_age_pred fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json


# Train the Contrastive Predictive Coding (CPC) model; inference
python ../../train_cpc.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json
for SC_EPOCH in 04 08 12 16 20 24 28
do
    python ../../ml_inference.py \
        model_path.model="models/cpc_checkpoints/cpc_model_${SC_EPOCH##+(0)}.pth" \
        output.path="data/cpc_embedding_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_params.json
done
# Fine tune the CPC model in supervised mode and save scores to the file
python -m scenario_age_pred fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_cpc_params.json


# Train the Contrastive Predictive Coding (CPC v2) model; inference 
python ../../cpc_v2_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_v2_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/cpc_v2_params.json


# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_age_pred compare_approaches --n_workers 1 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "mles_embeddings.pickle" "cpc_embedding*.pickle" "cpc_v2_embeddings.pickle" \
    --score_file_names "target_scores" "mles_finetuning_scores" 'cpc_finetuning_scores'


# check the results
cat results/scenario.csv
```

### Semi-supervised setup
```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"

# run semi supervised scenario
./bin/scenario_semi_supervised.sh

# check the results
cat results/semi_scenario_age_pred_*.csv

```

### Test model configurations

```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"

# run all scenarios or select one
./bin/*.sh

# check the results
cat results/scenario_age_pred_*.csv
```

### Transformer network
```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"

# Train the MeLES encoder on transformer and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/transformer_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/transformer_params.json

python -m scenario_age_pred fit_finetuning \
    params.device="$SC_DEVICE" \
    --conf conf/dataset.hocon conf/fit_finetuning_on_transf_params.json

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

# Train the encoder on transformer and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_proj_head_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_proj_head_params.json

# Check some options with `--help` argument
python -m scenario_age_pred compare_approaches --n_workers 3 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "mles_proj_head_embeddings.pickle"

```

# New baseline via AggFeatureModel

```sh
cd experiments/scenario_age_pred
export SC_DEVICE="cuda"

# Prepare agg feature encoder and take embedidngs; inference
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_params.json
python ../../ml_inference.py    params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_params.json

python -m scenario_age_pred compare_approaches --n_workers 5 \
    --add_baselines --add_emb_baselines \
    --embedding_file_names "agg_feat_embed.pickle"

```

# Complex Learning

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
