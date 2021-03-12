export SC_SUFFIX="SampleRandom"
export SC_STRATEGY="SampleRandom"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    params.device="$SC_DEVICE" \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

export SC_SUFFIX="SampleRandom_short"
export SC_STRATEGY="SampleRandom"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    data_module.train.max_seq_len=600 \
    data_module.valid.max_seq_len=600 \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# Compare
rm results/scenario_bowl2019__subseq_smpl_strategy.txt
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 --local_scheduler \
    --conf_extra \
      'report_file: "../results/scenario_bowl2019__subseq_smpl_strategy.txt",
      auto_features: [
          "../data/emb__SampleRandom.pickle",
          "../data/emb__SampleRandom_short.pickle"]'

