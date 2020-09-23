# AllPositivePair
export SC_SUFFIX="smpl_strategy_AllPositivePair"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="AllPositivePair" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# DistanceWeightedPair
export SC_SUFFIX="smpl_strategy_DistanceWeightedPair"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="DistanceWeightedPair" \
    params.train.n_samples_from_class=5 \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# HardNegativePair
for SC_NEG_COUNT in 2 5 9
do
  export SC_SUFFIX="smpl_strategy_HardNegativePair_neg_count_${SC_NEG_COUNT}"
  python ../../metric_learning.py \
      params.device="$SC_DEVICE" \
      params.train.sampling_strategy="HardNegativePair" \
      params.train.neg_count=${SC_NEG_COUNT} \
      model_path.model="models/mles__$SC_SUFFIX.p" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
  python ../../ml_inference.py \
      params.device="$SC_DEVICE" \
      model_path.model="models/mles__$SC_SUFFIX.p" \
      output.path="data/emb_mles__$SC_SUFFIX" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
done

# Compare
rm results/scenario_rosbank__smpl_strategy.txt
# rm -r conf/embeddings_validation.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_rosbank__smpl_strategy.txt",
      auto_features: ["../data/emb_mles__smpl_*.pickle"]'
