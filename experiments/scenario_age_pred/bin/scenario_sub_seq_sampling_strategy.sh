export SC_SUFFIX="SampleRandom"
export SC_STRATEGY="SampleRandom"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    params.train.split_strategy.cnt_min=200 \
    params.train.split_strategy.cnt_max=600 \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"


export SC_SUFFIX="SplitRandom"
export SC_STRATEGY="SplitRandom"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    data_module.train.max_seq_len=600 \
    data_module.valid.max_seq_len=600 \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# Compare
rm results/scenario_age_pred__subseq_smpl_strategy.txt

python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_age_pred__subseq_smpl_strategy.txt",
      auto_features: [
          "../data/emb__SplitRandom.pickle",
          "../data/emb__SampleRandom.pickle"]'
