export SC_SUFFIX="subseq_SampleRandom"
export SC_STRATEGY="SampleRandom"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

export SC_SUFFIX="subseq_SplitRandom"
export SC_STRATEGY="SplitRandom"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# Compare
rm results/scenario_rosbank__subseq_smpl_strategy.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_rosbank__subseq_smpl_strategy.txt",
      auto_features: ["../data/emb_mles__subseq_*.pickle"]'
