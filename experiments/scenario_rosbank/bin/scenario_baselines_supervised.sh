# Train a supervised model and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_target.hocon

# Fine tune the MeLES model in supervised mode and save scores to the file
python ../../pl_train_module.py \
    params.rnn.type="gru" params.rnn.hidden_size=512 params.train.n_epoch=50 \
    model_path="models/mles_model_for_finetuning.p" \
    --conf conf/mles_params.hocon

python ../../pl_fit_target.py \
    params.rnn.type="gru" params.rnn.hidden_size=512 \
    params.pretrained_model_path="models/mles_model_for_finetuning.p" \
    --conf conf/pl_fit_finetuning_mles.hocon

# Fine tune the CPC model in supervised mode and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_cpc.hocon

# Fine tune the NSP and RTD model in supervised mode and save scores to the file
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_nsp.hocon
python ../../pl_fit_target.py --conf conf/pl_fit_finetuning_rtd.hocon

# Compare
rm results/scenario_rosbank_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20
