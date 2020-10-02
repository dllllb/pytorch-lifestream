# Train a supervised model and save scores to the file
python -m scenario_bowl2019 fit_target params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_target_params.json

# Fine tune the MeLES model in supervised mode and save scores to the file
#cp models/mles_model.p models/mles_model_ft.p
python ../../metric_learning.py \
  params.device="$SC_DEVICE" \
  params.train.neg_count=5 \
  model_path.model="models/mles_model_ft.p" \
  --conf conf/trx_dataset.hocon conf/mles_params.json
python -m scenario_bowl2019 fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json

# Train a special CPC model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
python ../../train_cpc.py params.device="$SC_DEVICE" --conf conf/trx_dataset.hocon conf/cpc_params_for_finetuning.json
# Fine tune the CPC model in supervised mode and save scores to the file
python -m scenario_bowl2019 fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_cpc_params.json

# Fine tune the RTD model in supervised mode and save scores to the file
python -m scenario_bowl2019 fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_rtd_params.json

# Compare
rm results/scenario_bowl2019_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20
