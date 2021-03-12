# Train a supervised model and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_target.hocon

# Fine tune the MeLES model in supervised mode and save scores to the file
python ../../pl_train_module.py \
    params.train.neg_count=5 \
    model_path="models/mles_model_ft.p" \
    --conf conf/mles_params.hocon

python ../../pl_fit_target.py \
    params.pretrained.model_path="models/mles_model_ft.p" \
    data_module.train.drop_last=true \
    --conf conf/pl_fit_finetuning_mles.hocon

# Train a special CPC model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
python ../../pl_train_module.py --conf conf/cpc_params_for_finetuning.hocon

# Fine tune the CPC model in supervised mode and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_cpc.hocon

# Fine tune the RTD model in supervised mode and save scores to the file
python ../../pl_fit_target.py data_module.train.drop_last=true --conf conf/pl_fit_finetuning_rtd.hocon

# Compare
rm results/scenario_bowl2019_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20 --local_scheduler
