# Get data

```sh
cd experiments/scenario_x5

# download datasets
bin/get-data.sh

# convert datasets from transaction list to features for metric learning
bin/make-datasets-spark.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_x5
export SC_DEVICE="cuda"

sh bin/run_all_scenarios.sh

```

# Another baselines

```sh
# Train the Sequence Order Prediction (SOP) model; inference
python ../../train_sop.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/sop_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/sop_params.json

# Train the Next Sequence Prediction (NSP) model; inference
python ../../train_nsp.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/nsp_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/nsp_params.json

# Train the Replaced Token Detection (RTD) model; inference
python ../../train_rtd.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/rtd_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/rtd_params.json

# Train a special MeLES model for fine-tuning 
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model. 
python ../../train_rtd.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/rtd_params_for_finetuning.json
# Fine tune the RTD model in supervised mode and save scores to the file
python -m scenario_x5 fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_rtd_params.json

# Run estimation 
python -m scenario_x5 compare_approaches --n_workers 1 --models lgb \
    --output_file results/scenario_x5_baselines.csv \
    --embedding_file_names \
        "sop_embeddings.pickle" \
        "nsp_embeddings.pickle" \
        "rtd_embeddings.pickle" \
    --score_file_names  "rtd_finetuning_scores"
```