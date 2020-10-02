# Train a supervised model and save scores to the file
python -m scenario_age_pred fit_target params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_target_params.json

# Train a special MeLES model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
python ../../metric_learning.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params_for_finetuning.json
# Take the pretrained metric learning model and fine tune it in supervised mode; save scores to the file
python -m scenario_age_pred fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json

python -m scenario_age_pred fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_cpc_params.json

# Fine tune the RTD model in supervised mode and save scores to the file
python -m scenario_age_pred fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_rtd_params.json

# Compare
rm results/scenario_age_pred_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20
