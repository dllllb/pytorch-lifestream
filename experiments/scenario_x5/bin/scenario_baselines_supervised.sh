# Train a supervised model and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_target_rnn.hocon

# Train a special MeLES model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
python ../../pl_train_module.py --conf conf/mles_params_for_finetuning.hocon
# Fine tune the MeLES model in supervised mode and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_on_mles.hocon

# Fine tune the CPC model in supervised mode and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_on_cpc.hocon

# Train a special MeLES model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
python ../../pl_train_module.py --conf conf/rtd_params_for_finetuning.hocon
# Fine tune the RTD model in supervised mode and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_on_rtd.hocon

cp "models/barlow_twins_model.p" "models/barlow_twins_model_ft.p"
# lightning_logs/bt_tuning_base/version_0/checkpoints/epoch\=20-step\=23750.ckpt
#python ../../pl_train_module.py \
#  params.rnn.hidden_size=160 \
#  trainer.max_epochs=100 \
#  model_path="models/barlow_twins_model_for_finetuning.p" \
#  --conf conf/barlow_twins_params.hocon
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_on_barlow_twins.hocon

# Compare
rm results/scenario_x5_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20
