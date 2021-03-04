# AllPositivePair
export SC_SUFFIX="smpl_strategy_AllPositivePair"
python ../../pl_train_module.py \
  logger_name=${SC_SUFFIX} \
  params.train.sampling_strategy="AllPositivePair" \
  model_path="models/mles__$SC_SUFFIX.p" \
  --conf "conf/mles_params.hocon"

python ../../pl_inference.py \
  model_path="models/mles__$SC_SUFFIX.p" \
  output.path="data/emb__$SC_SUFFIX" \
  --conf "conf/mles_params.hocon"


# DistanceWeightedPair
export SC_SUFFIX="smpl_strategy_DistanceWeightedPair"
python ../../pl_train_module.py \
  logger_name=${SC_SUFFIX} \
  params.train.sampling_strategy="DistanceWeightedPair" \
  params.train.n_samples_from_class=5 \
  model_path="models/mles__$SC_SUFFIX.p" \
  --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# HardNegativePair
for SC_NEG_COUNT in 2 5 9
do
  export SC_SUFFIX="smpl_strategy_HardNegativePair_neg_count_${SC_NEG_COUNT}"
  python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.sampling_strategy="HardNegativePair" \
    params.train.neg_count=${SC_NEG_COUNT} \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
  python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"
done

# Compare
rm results/scenario_age_pred__smpl_strategy.txt

python -m embeddings_validation \
  --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
  --conf_extra \
    'report_file: "../results/scenario_age_pred__smpl_strategy.txt",
    auto_features: ["../data/emb__smpl_strategy_*.pickle"]'
