# Train a supervised model and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_target.hocon

# Fine tune the MeLES model in supervised mode and save scores to the file
python ../../pl_train_module.py \
  params.rnn.hidden_size=256 \
  params.train.loss="MarginLoss" params.train.margin=0.2 params.train.beta=0.4 \
  model_path="models/mles_model_for_finetuning.p" \
  --conf conf/mles_params.hocon

python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_mles.hocon

# Fine tune the CPC model in supervised mode and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_cpc.hocon

# Fine tune the RTD model in supervised mode and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_rtd.hocon

# Fine tune the MeLES model in supervised mode and save scores to the file
python ../../pl_train_module.py \
  params.rnn.hidden_size=256 \
  model_path="models/barlow_twins_model_for_finetuning.p" \
  --conf conf/mles_params.hocon
# Fine tune the RTD model in supervised mode and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_barlow_twins.hocon

# Compare
rm results/scenario_gender_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20
