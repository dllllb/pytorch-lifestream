export SC_SUFFIX="subseq_smpl_SampleRandom"
export SC_STRATEGY="SampleRandom"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"


export SC_SUFFIX="subseq_smpl_SplitRandom"
export SC_STRATEGY="SplitRandom"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    data_module.train.max_seq_len=300 \
    data_module.valid.max_seq_len=300 \
    data_module.valid.batch_size=512 \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"


# Compare
rm results/scenario_gender__subseq_smpl_strategy.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_gender__subseq_smpl_strategy.txt",
      auto_features: ["../data/emb__subseq_smpl_*.pickle"]'
