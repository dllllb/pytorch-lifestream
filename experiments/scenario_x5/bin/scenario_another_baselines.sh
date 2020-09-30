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

# Compare
rm results/scenario_x5_baselines.txt
# rm -r conf/embeddings_validation.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_another_baselines.hocon --workers 10 --total_cpu_count 20
